
import numpy as np
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.nn import Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader
)
from torch_geometric.nn import TGNMemory
from sklearn.metrics import average_precision_score, roc_auc_score
from itertools import combinations

import datetime
from src.config import *
from src.log.log_handler import get_logger
logger = get_logger(__name__)



class TemporalReachability:
    def __init__(self, edge_list):
        """
        Initializes the Temporal Reachability object.

        Parameters:
        - edge_list (list of tuples): List of edges as (src, dst) tuples.
        """
        self.graph = defaultdict(list)

        for src, dst in edge_list:
            self.graph[src].append(dst)

    def calculate_source_depth(self, source_node):
        """
        Calculate the depth of the source node by finding how deep it connects to other nodes.

        Parameters:
        - source_node (int): The source node to calculate depth for.

        Returns:
        - depth (int): The maximum depth of connections originating from the source node.
        """
        visited = set()
        queue = [(source_node, 0)]  # (current_node, depth)
        visited.add(source_node)
        max_depth = 0

        while queue:
            current_node, depth = queue.pop(0)

            # Update max depth
            max_depth = max(max_depth, depth)

            # Traverse neighbors
            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return max_depth
    
    def print_all_paths(self):
        """
        Prints all paths in the graph for debugging.
        """
        for src, neighbors in self.graph.items():
            for dst in neighbors:
                logger.info(f"Edge: {src} -> {dst}")

class EdgeMemory:
    def __init__(self, memory_dim):
        self.memory_dim = memory_dim
        self.edge_memory = {}  # Dictionary to store current memory for each edge
        self.edge_history = {}  # Dictionary to store historical memory for each edge

    def initialize_edge(self, edge_id):
        """Initialize memory for a new edge."""
        if edge_id not in self.edge_memory:
            self.edge_memory[edge_id] = torch.zeros(self.memory_dim, requires_grad=True)
            self.edge_history[edge_id] = []

    def update_memory(self, edge_id, new_message, aggregator=None):
        self.initialize_edge(edge_id)
        past_memory = self.edge_memory[edge_id]

        # Ensure both tensors are 2D
        if past_memory.ndimension() == 1:
            past_memory = past_memory.unsqueeze(0)
        if new_message.ndimension() == 1:
            new_message = new_message.unsqueeze(0)

        # Update logic with aggregator
        if aggregator:
            updated_memory = aggregator(past_memory, new_message)
        else:  # Default: weighted average
            updated_memory = 0.5 * past_memory + 0.5 * new_message

        self.edge_memory[edge_id] = updated_memory
        self.edge_history[edge_id].append(updated_memory.clone())


    def get_memory(self, edge_id):
        memory = self.edge_memory.get(edge_id, torch.zeros(self.memory_dim))
        if memory.ndimension() == 1:  # Ensure the tensor is 2D
            memory = memory.unsqueeze(0)  # Shape: [1, memory_dim]
        return memory

    def get_history(self, edge_id):
        """Retrieve historical memory for an edge."""
        return self.edge_history.get(edge_id, [])

class AttentionAggregator(torch.nn.Module):
    def __init__(self, memory_dim, msg_dim):
        super().__init__()
        self.linear = Linear(memory_dim + msg_dim, memory_dim)

    def forward(self, past_memory, new_message):
        # Ensure both tensors are 2D
        if past_memory.ndimension() == 1:
            past_memory = past_memory.unsqueeze(0)  # Shape: [1, memory_dim]
        if new_message.ndimension() == 1:
            new_message = new_message.unsqueeze(0)  # Shape: [1, msg_dim]

        # Concatenate tensors
        combined = torch.cat([past_memory, new_message], dim=-1)

        # Pass through Linear layer
        try:
            output = self.linear(combined).relu()
            return output
        except Exception as e:
            logger.info(f"Error during Linear layer forward pass: {e}")
            raise


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_heads=8):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // num_heads, heads=num_heads,
            dropout=0.3, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        # Compute relative time encodings
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        # Apply causal mask: Only allow messages from the past
        causal_mask = (rel_t >= 0).float().unsqueeze(1)  # Shape: [num_edges, 1]
        edge_attr = edge_attr * causal_mask

        return self.conv(x, edge_index, edge_attr)

class GraphAttentionEmbedding_mem(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, edge_memory, num_heads=8, memory_dim=128):
        super().__init__()
        self.time_enc = time_enc
        self.memory_dim = memory_dim
        self.edge_memory = edge_memory  # Pass edge memory instance
        edge_dim = msg_dim + time_enc.out_channels + memory_dim
        self.conv = TransformerConv(
            in_channels, out_channels // num_heads, heads=num_heads,
            dropout=0.3, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        # Compute relative time encodings
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        
        # Retrieve and concatenate edge memory
        edge_ids = [(src.item(), dst.item()) for src, dst in edge_index.t()]
        edge_mem_list = [self.edge_memory.get_memory(edge_id) for edge_id in edge_ids]
        # Ensure all tensors in edge_mem_list are 2D before concatenation
        if len(edge_mem_list) > 0:
            edge_mem = torch.cat(edge_mem_list, dim=0)  # Concatenate along the batch dimension
        else:
            edge_mem = torch.zeros((0, self.memory_dim)).to(x.device)  # Handle empty case
            
        #edge_mem = torch.cat(edge_mem_list, dim=0) if edge_mem_list else torch.zeros((0, self.memory_dim)).to(x.device)

        # Concatenate memory with relative time encodings and messages
        edge_attr = torch.cat([rel_t_enc, msg, edge_mem], dim=-1)

        # Apply causal mask
        causal_mask = (rel_t >= 0).float().unsqueeze(1)
        edge_attr = edge_attr * causal_mask

        return self.conv(x, edge_index, edge_attr)

    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
        #return torch.sigmoid(self.lin_final(h))

class TGNModel:
    def __init__(self, data, device, memory_dim=128, time_dim=128, embedding_dim=128, num_heads=8):
        self.device = device
        self.data = data.to(device)
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.num_nodes = data.num_nodes
        self.node_feature_dim = data.x.size(-1)
        self.num_heads = num_heads

        # Instantiate EdgeMemory and AttentionAggregator
        #self.aggregator = AttentionAggregator(memory_dim, memory_dim+data.msg.size(-1)).to(device)
        #self.edge_memory = EdgeMemory(memory_dim)  # Add edge memory

        # Edge history storage
        self.edge_history = defaultdict(list)  # {(src, dst): [weights_over_time]}

        self.memory = None
        self.initialize_memory()

        self.gnn = GraphAttentionEmbedding(
            in_channels=self.node_feature_dim,
            out_channels=self.embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=self.memory.time_enc,
            #edge_memory=self.edge_memory,
            num_heads=self.num_heads
        ).to(device)

        self.link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

        self.optimizer = torch.optim.AdamW(
            set(self.memory.parameters()) | set(self.gnn.parameters()) | set(self.link_pred.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
        self.neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)


        # Initialize temporal reachability
        edge_list = list(zip(data.src.cpu().tolist(), data.dst.cpu().tolist()))
        self.temporal_reachability = TemporalReachability(edge_list)

        self.model_loader_ind = False

    def get_source_depth(self, source_node):
        """
        Wrapper to calculate the depth of a source node.

        Parameters:
        - source_node (int): The source node to calculate depth for.

        Returns:
        - depth (int): The maximum depth of connections originating from the source node.
        """
        return self.temporal_reachability.calculate_source_depth(source_node)

    def print_all_paths(self):
        """
        Wrapper to print all paths in the graph.
        """
        self.temporal_reachability.print_all_paths()

    def update_edge_history(self, src, dst, weight):
        """Update historical weights for an edge."""
        edge_key = (src.item(), dst.item())
        new_weight = float(weight.cpu().detach().numpy()[0])
        self.edge_history[edge_key].append(new_weight)


    def get_edge_history(self, src, dst):
        """Retrieve historical weights for an edge."""
        edge_key = (src, dst)
        return self.edge_history.get(edge_key, [])
    
    def initialize_optimizer(self):
        # Add debug prints to see parameter shapes
        #print("Parameter shapes:")
        #for name, param in self.gnn.named_parameters():
        #    print(f"{name}: {param.shape}")
        
        # Initialize optimizer with correct parameter groups
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.memory.parameters()},
                {'params': self.gnn.parameters()},
                {'params': self.link_pred.parameters()}
            ],
            lr=0.001,
            weight_decay=1e-5
        )

    def initialize_memory(self):
        """Initialize memory with current number of nodes"""
        self.memory = TGNMemory(
            self.num_nodes,
            self.data.msg.size(-1),
            self.memory_dim,
            self.time_dim,
            message_module=IdentityMessage(self.data.msg.size(-1), self.memory_dim, self.time_dim),
            aggregator_module=LastAggregator(),
        ).to(self.device)

    def train_executor(self, data_module, num_epochs, best_model_path):
        # Data Loaders
        train_loader = data_module.train_loader
        val_loader = data_module.val_loader
        #train_data = data_module.train_data

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=1, factor=0.1, verbose=True
        )

        best_val_auc = 0.0

        # Training Loop
        for epoch in range(1, num_epochs + 1):
            loss = self.train_one_epoch(train_loader)
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

            val_ap, val_auc = self.evaluate(val_loader)
            logger.info(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')

            scheduler.step(loss)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.save_model(epoch=epoch, path=best_model_path)
                #logger.info(f'Best model saved with AUC: {best_val_auc:.4f}')


                
    def train_one_epoch_mem(self, train_loader):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()
        self.neighbor_loader.reset_state()

        total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z_memory, last_update = self.memory(n_id)

            x = self.data.x[n_id].to(self.device)
            z = self.gnn(
                x, last_update, edge_index,
                self.data.t[e_id].to(self.device), self.data.msg[e_id].to(self.device)
            )

            for src, dst, t, msg in zip(batch.src, batch.dst, batch.t, batch.msg):
                edge_id = (src.item(), dst.item())
                edge_mem = self.edge_memory.get_memory(edge_id)
                # Ensure both are 2D
                if msg.ndimension() == 1:
                    msg = msg.unsqueeze(0)
                if edge_mem.ndimension() == 1:
                    edge_mem = edge_mem.unsqueeze(0)

                # Concatenate memory and message
                message = torch.cat([edge_mem, msg], dim=-1)

                # Update edge memory using aggregator
                self.edge_memory.update_memory(edge_id, message, aggregator=self.aggregator)

            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            loss = self.criterion(pos_out, torch.ones_like(pos_out))
            loss += self.criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.memory.detach()
            total_loss += float(loss) * batch.num_events

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return total_loss / len(train_loader.dataset)



    def train_one_epoch(self, train_loader):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()
        self.neighbor_loader.reset_state()

        total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z_memory, last_update = self.memory(n_id)

            x = self.data.x[n_id].to(self.device)
            z = self.gnn(
                x, last_update, edge_index,
                self.data.t[e_id].to(self.device), self.data.msg[e_id].to(self.device)
            )

            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            # Update historical weights
            for src, dst, weight in zip(batch.src, batch.dst, pos_out):
                self.update_edge_history(src, dst, weight)
                
            loss = self.criterion(pos_out, torch.ones_like(pos_out))
            loss += self.criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            self.optimizer.step()
            self.memory.detach()
            total_loss += float(loss) * batch.num_events

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        torch.manual_seed(12345)

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z_memory, last_update = self.memory(n_id)
            x = self.data.x[n_id].to(self.device)
            z = self.gnn(
                x, last_update, edge_index,
                self.data.t[e_id].to(self.device), self.data.msg[e_id].to(self.device)
            )

            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                 torch.zeros(neg_out.size(0))], dim=0
            )

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

    def save_model(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'memory_state_dict': self.memory.state_dict(),
            'gnn_state_dict': self.gnn.state_dict(),
            'link_pred_state_dict': self.link_pred.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_nodes': self.num_nodes,
            'edge_history': self.edge_history  # Save the historical edge weights
        }, path)

    def load_model(self, path):
        if self.model_loader_ind:
            logger.info(f"TGN model already loaded at epoch {self.epoch} in {path}")
            return self.epoch

        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        trained_num_nodes = checkpoint['num_nodes']
        self.num_nodes = max(trained_num_nodes, self.num_nodes)

        # Reinitialize memory with correct size
        self.initialize_memory()

        

        # Load state dict and expand tensors if needed
        memory_state = checkpoint['memory_state_dict']
        if  self.num_nodes > trained_num_nodes:
            # Expand memory tensor
            old_memory = memory_state['memory']
            new_memory = torch.zeros(( self.num_nodes, old_memory.size(1)))
            new_memory[:trained_num_nodes] = old_memory
            memory_state['memory'] = new_memory

            # Expand last_update tensor
            old_last_update = memory_state['last_update']
            new_last_update = torch.zeros( self.num_nodes)
            new_last_update[:trained_num_nodes] = old_last_update
            memory_state['last_update'] = new_last_update

            # Expand _assoc tensor
            old_assoc = memory_state['_assoc']
            new_assoc = torch.zeros( self.num_nodes)
            new_assoc[:trained_num_nodes] = old_assoc
            memory_state['_assoc'] = new_assoc


        # Load expanded state dictionaries
        self.memory.load_state_dict(memory_state)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.link_pred.load_state_dict(checkpoint['link_pred_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.initialize_optimizer()

        # Update assoc and neighbor_loader
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=self.device)
        self.neighbor_loader = LastNeighborLoader(self.num_nodes, size=10, device=self.device)

        # Restore the edge history
        if 'edge_history' in checkpoint:
            self.edge_history = checkpoint['edge_history']
        else:
            logger.info("Warning: No edge history found in checkpoint. Initializing as empty.")
            self.edge_history = defaultdict(list)

        self.epoch = checkpoint['epoch']

        self.model_loader_ind = True

        #logger.info(f"TGN model loaded at epoch {self.epoch} in {path}")
        return self.epoch


    @torch.no_grad()
    def inference_executor(self, data, model_path=MODEL_DIR+"/model.pth"):
        

        # Load the trained model
        self.load_model(path=model_path)

        # Prepare Data Loader for inference (load all data without splitting)
        inference_loader = TemporalDataLoader(
            data,
            batch_size=140,
            neg_sampling_ratio=1.0,
            shuffle=False
        )

        # Set the model to evaluation mode
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        # Initialize neighbor loader and assoc tensor for the inference data
        self.memory.reset_state()
        num_nodes = self.memory.num_nodes  # Ensure this matches the training model
        self.neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=self.device)
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=self.device)


        # Evaluate the model on the entire dataset
        #test_ap, test_auc = self.evaluate_inference(inference_loader, data)
        #print(f'Inference AP: {test_ap:.4f}, Inference AUC: {test_auc:.4f}')

        # Extract embeddings from the model
        node_embeddings, link_embeddings, edge_list = self.extract_embeddings(inference_loader, data)

        return node_embeddings, link_embeddings, edge_list

    @torch.no_grad()
    def evaluate_inference(self, loader, data):
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        torch.manual_seed(12345)

        aps, aucs = [], []
        for batch in loader:
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z_memory, last_update = self.memory(n_id)
            x = data.x[n_id].to(self.device)
            z = self.gnn(
                x, last_update, edge_index,
                data.t[e_id].to(self.device), data.msg[e_id].to(self.device)
            )

            pos_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.dst]])
            neg_out = self.link_pred(z[self.assoc[batch.src]], z[self.assoc[batch.neg_dst]])

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                 torch.zeros(neg_out.size(0))], dim=0
            )

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())
    
    @torch.no_grad()
    def extract_embeddings(self, loader, data):
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        self.memory.reset_state()
        self.neighbor_loader.reset_state()

        node_embeddings = {}
        link_embeddings = {}
        edge_list = []

        for batch in loader:
            batch = batch.to(self.device)

            n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            z_memory, last_update = self.memory(n_id)
            x = self.data.x[n_id].to(self.device)
            z = self.gnn(
                x, last_update, edge_index,
                data.t[e_id].to(self.device), data.msg[e_id].to(self.device)
            )

            # Compute predictions for the current batch
            src_embeds = z[self.assoc[batch.src]]
            dst_embeds = z[self.assoc[batch.dst]]
            current_predictions = self.link_pred(src_embeds, dst_embeds).cpu()

            for i, (src, dst) in enumerate(zip(batch.src, batch.dst)):
                key = f"{src.item()}_{dst.item()}"
                edge_list.append((src.item(), dst.item()))
                node_embeddings[src.item()] = z[self.assoc[src]].cpu().numpy()
                node_embeddings[dst.item()] = z[self.assoc[dst]].cpu().numpy()

                # Retrieve historical weights for this edge
                historical_weights = self.get_edge_history(src.item(), dst.item())

                # Append the latest prediction to historical weights
                updated_weights = historical_weights + [float(current_predictions[i].cpu().numpy())]
                #updated_weights = np.array(historical_weights + [current_predictions[i].cpu().numpy()], dtype=np.float32)
                link_embeddings[key] = updated_weights

            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            self.neighbor_loader.insert(batch.src, batch.dst)

        return node_embeddings, link_embeddings, edge_list


    def calculate_key_metrics(self, edge_list=None):
        # Track edge frequencies
        edge_frequency = {}
        for edge in self.data.edge_index.cpu().numpy().T:
            src, dst = sorted(edge)
            pair = (src,dst)
            if pair in edge_frequency:
                edge_frequency[pair] += 1
            else:
                edge_frequency[pair] = 1

        metrics_dict = {}
        num_nodes = self.data.num_nodes

        # Create adjacency matrix for other metrics
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for edge in self.data.edge_index.cpu().numpy().T:
            adjacency_matrix[edge[0], edge[1]] = 1
            adjacency_matrix[edge[1], edge[0]] = 1

        # Determine node pairs to calculate metrics for
        if edge_list is not None:
            pairs_to_evaluate = edge_list
        else:
            pairs_to_evaluate = list(combinations(range(num_nodes), 2))

        for src, dst in pairs_to_evaluate:
            pair_key = f"{src}_{dst}"

            # Metric 1: Edge Count (Use frequency dictionary)
            edge_count = edge_frequency.get((min(src, dst), max(src, dst)), 0)

            # Other metrics (use adjacency_matrix as before)
            neighbors_src = set(np.where(adjacency_matrix[src] == 1)[0])
            neighbors_dst = set(np.where(adjacency_matrix[dst] == 1)[0])

            common_neighbors = len(neighbors_src & neighbors_dst)
            union_neighbors = len(neighbors_src | neighbors_dst)
            jaccard_similarity = len(neighbors_src & neighbors_dst) / union_neighbors if union_neighbors > 0 else 0
            adamic_adar = sum(1 / np.log(len(np.where(adjacency_matrix[k] == 1)[0]))
                            for k in neighbors_src & neighbors_dst if len(np.where(adjacency_matrix[k] == 1)[0]) > 1)
            preferential_attachment = len(neighbors_src) * len(neighbors_dst)

            metrics = [edge_count, common_neighbors, jaccard_similarity, adamic_adar, preferential_attachment]
            if any(metric > 0 for metric in metrics):
                metrics_dict[pair_key] = metrics

        sorted_metrics_dict = dict(sorted(metrics_dict.items(), key=lambda x: sum(x[1]), reverse=True))
        return sorted_metrics_dict


    def calculate_edge_metrics(self, edge_list, node_embeddings, link_predictions):
        """
        Calculate edge metrics for a given edge list using node embeddings and link predictions.

        Parameters:
        - edge_list (list of tuples): List of edges (src, dst).
        - node_embeddings (dict): Node embeddings from TGN as {node_id: embedding}.
        - link_predictions (dict): Link predictions as {(src, dst): predicted_score}.
        - num_nodes (int): Total number of nodes.

        Returns:
        - edge_metrics (dict): Dictionary where keys are 'src_dst' strings and values are lists of metrics.
        """
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Create graph from edge list
        G = nx.Graph()
        G.add_edges_from(edge_list)

        # Initialize metrics dictionary
        edge_metrics = {}

        for src, dst in edge_list:
            # Common Neighbors
            neighbors_src = set(G.neighbors(src))
            neighbors_dst = set(G.neighbors(dst))
            common_neighbors = len(neighbors_src & neighbors_dst)

            # Jaccard Similarity
            union_neighbors = len(neighbors_src | neighbors_dst)
            jaccard_similarity = len(neighbors_src & neighbors_dst) / union_neighbors if union_neighbors > 0 else 0

            # Adamic-Adar Index
            adamic_adar_index = sum(1 / np.log(len(list(G.neighbors(w)))) for w in neighbors_src & neighbors_dst if len(list(G.neighbors(w))) > 1)

            # Preferential Attachment
            preferential_attachment = len(neighbors_src) * len(neighbors_dst)

            # Embedding Similarity (Cosine)
            if src in node_embeddings and dst in node_embeddings:
                src_emb = node_embeddings[src]
                dst_emb = node_embeddings[dst]
                embedding_similarity = cosine_similarity([src_emb], [dst_emb])[0, 0]
            else:
                embedding_similarity = 0

            # Link Prediction Score
            #link_prediction_score = link_predictions.get((src, dst), link_predictions.get((dst, src), 0))

            new_depth=self.get_source_depth(src)

            # Create the key and store metrics
            edge_key = f"{src}_{dst}"
            edge_metrics[edge_key] = [
                common_neighbors,
                jaccard_similarity,
                adamic_adar_index,
                preferential_attachment,
                embedding_similarity,
                new_depth
            ]

        return edge_metrics



class TGNDataModule:
    def __init__(self, nodes_df, edges_df):#, record_id_to_idx=None):
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        self.data = None  # Will hold the TemporalData object
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.neighbor_loader = None

    def save_temporal_data(self, timestamp,mode,edges_df):
        """Save TemporalData object to disk"""
        torch.save(self.data, f"{MODEL_DIR}/{os.getenv('DATASET')}/data_{mode}_{timestamp.replace(':','_')}.pt")
        edges_df.to_csv(f"{MODEL_DIR}/{os.getenv('DATASET')}/edges_{mode}_{timestamp.replace(':','_')}.csv", index=False)

    def load_temporal_data(self, filepath):
        """Load TemporalData object from disk"""
        self.data = torch.load(filepath)

    def preprocess(self):
        """
        Preprocesses the data: normalizes features, handles NaN, prepares node features,
        ensures correct data types, and creates the TemporalData object.
        """
        # Step 1: Normalize the features (nodes and edge)
        # List of columns to normalize
        node_feature_cols = ['CpuUsageRate(%)', 'MemoryUsageRate(%)', 'AlarmCount', 'NetworkP90Latency(ms)','IsRoot','Depth','RecordRatio','ErrorRatio']
        #edge_feature_cols = ['weight']

        # Ensure consistent data types for record_id columns
        self.nodes_df['record_id'] = self.nodes_df['record_id'].astype(int)
        self.edges_df['source_record_id'] = self.edges_df['source_record_id'].astype(int)
        self.edges_df['target_record_id'] = self.edges_df['target_record_id'].astype(int)

        node_scaler = StandardScaler()
        #edge_scaler = StandardScaler()

        # Normalize node features
        for col in node_feature_cols:
            if col in self.nodes_df.columns:
                # Reshape to a 2D array for StandardScaler
                feature_values = self.nodes_df[col].values.reshape(-1, 1)

                # Fit and transform using StandardScaler
                self.nodes_df[col] = node_scaler.fit_transform(feature_values)

        # Normalize edge features
        #for col in edge_feature_cols:
        #    if col in self.edges_df.columns:
        #        # Reshape to a 2D array for StandardScaler
        #        feature_values = self.edges_df[col].values.reshape(-1, 1)
#
        #        # Fit and transform using StandardScaler
        #        self.edges_df[col] = edge_scaler.fit_transform(feature_values)

        # Step 2: Handle NaN values by replacing them with 0.0
        self.nodes_df.fillna(0.0, inplace=True)
        self.edges_df.fillna(0.0, inplace=True)

        self.record_ids = pd.concat([
            self.edges_df['source_record_id'],
            self.edges_df['target_record_id']
        ]).unique()

        # Prepare node features array
        node_features_list = []
        for record_id in self.record_ids:
            node_data = self.nodes_df[self.nodes_df['record_id'] == record_id]
            if not node_data.empty:
                features = node_data[['CpuUsageRate(%)', 'MemoryUsageRate(%)', 'AlarmCount', 'NetworkP90Latency(ms)','IsRoot','Depth','RecordRatio','ErrorRatio']].values[0]
            else:
                features = np.array([0.0] * 8)  # Default features if not found
                logger.info('missing feature for record_id:',record_id, ' node data:',node_data)
            node_features_list.append(features)
        #node_features = np.array(node_features_list, dtype=np.float32)

        # Step 4: Ensure data types
        #edge_index = torch.tensor(self.edges_df[['source_record_id', 'target_record_id']].values.T, dtype=torch.long)
        edge_weight = torch.tensor(self.edges_df['weight'].values, dtype=torch.float32)
        #timestamps = torch.tensor(self.edges_df['timestamp'].values, dtype=torch.long)


        # Step 5: Prepare traceid_hashed and mapping
        def hash_traceid(traceid):
            return hash(traceid) % (2**32)  # Hash to a 32-bit integer

        traceid_hashed = self.edges_df['TraceID'].apply(hash_traceid).values
        self.traceid_mapping = dict(zip(traceid_hashed, self.edges_df['TraceID']))

        # Create node features array that matches the size of num_nodes
        max_node_id = max(self.edges_df['source_record_id'].max(), self.edges_df['target_record_id'].max()) + 1
        full_node_features = np.zeros((max_node_id, len(node_feature_cols)), dtype=np.float32)

        # Fill in the features for nodes we have data for
        for record_id in self.record_ids:
            node_data = self.nodes_df[self.nodes_df['record_id'] == record_id]
            if not node_data.empty:
                features = node_data[node_feature_cols].values[0]
            else:
                features = np.zeros(len(node_feature_cols))
            full_node_features[record_id] = features

        # Create the TemporalData object
        self.data = TemporalData(
            src=torch.tensor(self.edges_df['source_record_id'].values, dtype=torch.long),
            dst=torch.tensor(self.edges_df['target_record_id'].values, dtype=torch.long),
            t=torch.tensor(self.edges_df['timestamp'].values, dtype=torch.long),
            msg=edge_weight.unsqueeze(1),
            x=torch.tensor(full_node_features, dtype=torch.float32),
            num_nodes=torch.tensor([max_node_id], dtype=torch.long)
        )

        # Set 'traceid' as an attribute
        self.data.traceid = torch.tensor(traceid_hashed, dtype=torch.long)
        




    def split_dataset(self,mode):
        """
        Splits the dataset into training, validation, and test sets based on TraceID,
        ensuring that all data from a given TraceID is in the same split.
        """
        # Step 1: Group data by traceid and preserve order
        traceid_to_indices = defaultdict(list)
        for idx in range(len(self.data.src)):
            traceid = self.data.traceid[idx].item()
            traceid_to_indices[traceid].append(idx)

        # Step 2: Get the traceids in the order they appear
        traceids = list(traceid_to_indices.keys())

        # Step 3: Split traceids into train, val, and test sets while preserving order
        total_traceids = len(traceids)

        if mode=='train_val':
            train_size = int(0.8 * total_traceids)
            val_size = total_traceids - train_size
            
            #train_traceids = traceids[:train_size]
            #val_traceids = traceids[train_size:train_size + val_size]

            val_traceids = traceids[:val_size]
            train_traceids = traceids[val_size:val_size+train_size]
            

            # Step 4: Collect indices for each split while preserving order
            train_indices = [idx for traceid in train_traceids for idx in traceid_to_indices[traceid]]
            val_indices = [idx for traceid in val_traceids for idx in traceid_to_indices[traceid]]
            self.train_data = self.data[train_indices]
            self.val_data = self.data[val_indices]
            self.train_data.traceid = self.data.traceid[train_indices]
            self.val_data.traceid = self.data.traceid[val_indices]

        elif mode=='train':
            train_size = int(1.0 * total_traceids)
            train_traceids = traceids[:train_size]
            train_indices = [idx for traceid in train_traceids for idx in traceid_to_indices[traceid]]
            self.train_data = self.data[train_indices]
            self.train_data.traceid = self.data.traceid[train_indices]


    def get_loaders(self,mode, batch_size=128, device='cpu'):
        """
        Creates TemporalDataLoaders for training, validation, and test sets.
        """
        if mode=='train_val':
            self.train_loader = TemporalDataLoader(
                self.train_data,
                batch_size=batch_size,
                neg_sampling_ratio=1.0,
                shuffle=True
            )
            self.val_loader = TemporalDataLoader(
                self.val_data,
                batch_size=batch_size,
                neg_sampling_ratio=1.0,
                shuffle=False
            )
        elif mode=='train':
             self.train_loader = TemporalDataLoader(
                self.train_data,
                batch_size=batch_size,
                neg_sampling_ratio=1.0,
                shuffle=True
            )           

        self.neighbor_loader = LastNeighborLoader(self.data.num_nodes, size=10, device=device)




def tgn_train_with_optimization(nodes_df, edges_df, num_epochs=DEFAULT_EPOCHS, base_model_path=MODEL_DIR):
    """
    Trains a TGN model and optimizes it by selecting the best parameters based on AUC score.

    Parameters:
    - nodes_df (DataFrame): Dataframe containing node features.
    - edges_df (DataFrame): Dataframe containing edge features and labels.
    - param_grid (dict): Dictionary defining the parameter grid for optimization.
    - num_epochs (int): Number of training epochs for each parameter set.
    - base_model_path (str): Base path to save the best and temporary models.

    Returns:
    - best_params (dict): Best parameter set based on AUC score.
    - best_auc (float): Best AUC score achieved.
    """
    from sklearn.model_selection import ParameterGrid
    import os


    # Example usage
    param_grid = {
        'memory_dim': [64, 128],
        'time_dim': [64, 128],
        'embedding_dim': [64, 128],
        'num_heads': [4, 8],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256],
        'dropout': [0.1, 0.3, 0.5],
        'neg_sampling_ratio': [1.0, 2.0, 5.0],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'scheduler_factor': [0.1, 0.5],
        'scheduler_patience': [1, 3, 5],
    }

    logger.info("TGN training with optimization started")

    # Initialize the data module
    data_mode = 'train_val'
    data_module = TGNDataModule(nodes_df, edges_df)

    # Preprocess the data
    data_module.preprocess()

    # Split the dataset into train and validation sets
    data_module.split_dataset(data_mode)

    # Get the data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_module.get_loaders(data_mode, batch_size=128, device=device)

    data = data_module.data

    # Parameter optimization setup
    param_combinations = list(ParameterGrid(param_grid))
    best_auc = 0.0
    best_params = None

    for params in param_combinations:
        logger.info(f"Testing parameters: {params}")

        # Initialize the model with the current parameters
        tgn_model = TGNModel(
            data,
            device,
            memory_dim=params['memory_dim'],
            time_dim=params['time_dim'],
            embedding_dim=params['embedding_dim'],
            num_heads=params['num_heads']
        )

        # Update optimizer and scheduler based on parameters
        tgn_model.optimizer = torch.optim.AdamW(
            set(tgn_model.memory.parameters()) | set(tgn_model.gnn.parameters()) | set(tgn_model.link_pred.parameters()),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            tgn_model.optimizer,
            mode='min',
            factor=params['scheduler_factor'],
            patience=params['scheduler_patience'],
            verbose=True
        )

        # Adjust batch size dynamically
        data_module.get_loaders(data_mode, batch_size=params['batch_size'], device=device)

        # Train the model
        tgn_model.train_executor(
            data_module,
            num_epochs=num_epochs,
            best_model_path=os.path.join(base_model_path, f"temp_model_{params}.pth")
        )

        # Evaluate the model
        _, val_auc = tgn_model.evaluate(data_module.val_loader)

        logger.info(f"Validation AUC for parameters {params}: {val_auc}")

        # Check if this is the best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params

            # Save the best model
            best_model_file = os.path.join(base_model_path, "model.pth")
            tgn_model.save_model(epoch=num_epochs, path=best_model_file)
            logger.info(f"New best model saved with AUC: {best_auc}")

    logger.info(f"Optimization completed. Best AUC: {best_auc} with parameters: {best_params}")
    logger.info(f"TGN Optimization Completed")
    raise Exception("TGN Optimization Completed")
    #return best_params, best_auc



def convert_timestamp_to_date(timestamp):
    from datetime import datetime
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d")

#TGN Train/Inference
def tgn_train(nodes_df, edges_df,train_epoch, timestamp):
    """
    Prepares data for TGN training using nodes_df and edges_df.
    """

    logger.info("TGN training started")
    data_mode='train_val'
    # Initialize the data module
    data_module = TGNDataModule(nodes_df, edges_df)

    # Preprocess the data
    data_module.preprocess()

    data_module.save_temporal_data(timestamp,data_mode,edges_df)

    # Split the dataset into train, val, and test sets
    data_module.split_dataset(data_mode)

    # Get the data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_module.get_loaders(data_mode,batch_size=140, device=device)

    data = data_module.data

    tgn_model = TGNModel(data, device)

    

    model_path=f'{MODEL_DIR}/{os.getenv("DATASET")}/model_{convert_timestamp_to_date(timestamp)}.pth'

    start_time=datetime.datetime.now()

    tgn_model.train_executor(data_module, num_epochs=train_epoch, best_model_path=model_path)

    logger.warning("trgn train duration: {}".format(datetime.datetime.now() - start_time))



def tgn_fine_tune(nodes_df, edges_df,timestamp):
    """
    Prepares data for TGN training using nodes_df and edges_df.
    """
    data_mode='train'
    # Initialize the data module
    data_module = TGNDataModule(nodes_df, edges_df)

    # Preprocess the data
    data_module.preprocess()

    # Split the dataset into train, val, and test sets
    data_module.split_dataset(data_mode)

    # Get the data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_module.get_loaders(data_mode,batch_size=140, device=device)

    data = data_module.data

    logger.info('\n\n#########################################')
    logger.info('Fine Tuning TGN started')
    tgn_model = TGNModel(data, device)

    model_path=f"{MODEL_DIR}/{os.getenv('DATASET')}/model_{timestamp}.pth"

    tgn_model.fine_tune_executor(data_module, num_epochs=DEFAULT_EPOCHS, model_path=model_path)

    logger.info('Fine Tuning TGN completed')
    logger.info('\n\n#########################################')


def tgn_inference(nodes_df, edges_df,mode, timestamp):
    """
    Performs TGN inference using nodes_df and edges_df.
    """
    logger.info('TGN inference started')
    # Load the mapping
    #with open('record_id_to_idx.pkl', 'rb') as f:
    #    record_id_to_idx = pickle.load(f)

    # Initialize the data module with the loaded mapping
    data_module = TGNDataModule(nodes_df, edges_df)#, record_id_to_idx=record_id_to_idx)

    # Preprocess the data
    data_module.preprocess()

    if mode == "inference":
        data_module.save_temporal_data(timestamp,mode,edges_df)

    # Get the data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data_module.data

    #print(f"Total interactions: {len(data.src)}")

    # Initialize the model
    tgn_model = TGNModel(data, device)

    # Set the idx_to_record_id mapping in the model
    #tgn_model.idx_to_record_id = data_module.idx_to_record_id

    # Perform inference
    model_path=f'{MODEL_DIR}/{os.getenv("DATASET")}/model_{convert_timestamp_to_date(timestamp)}.pth'

    node_embeddings, link_embeddings_norm, edge_list = tgn_model.inference_executor(data, model_path=model_path)

    # Calculate key metrics
    metrics_dict = tgn_model.calculate_edge_metrics(edge_list, node_embeddings, link_embeddings_norm)
    logger.info('TGN inference completed.')
    return node_embeddings, link_embeddings_norm, edge_list,metrics_dict
