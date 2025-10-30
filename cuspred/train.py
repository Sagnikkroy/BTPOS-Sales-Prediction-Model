import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Data Preparation and Feature Engineering
class RestaurantDataset(Dataset):
    def __init__(self, date_summary_path, date_seasons_path, date_item_path, seq_len=30, horizon=7):
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Load and merge datasets
        self.df = self.load_and_preprocess_data(date_summary_path, date_seasons_path, date_item_path)
        self.features, self.targets_sales, self.targets_orders = self.create_sequences()
        
    def load_and_preprocess_data(self, summary_path, seasons_path, item_path):
        # Load date_summary.csv
        df_summary = pd.read_csv(summary_path, names=['ds', 'sales', 'num_orders', 'avg_order_value'])
        df_summary['ds'] = pd.to_datetime(df_summary['ds'])
        
        # Load seasonal data
        df_seasons = pd.read_csv(seasons_path, names=['ds', 'season'])
        df_seasons['ds'] = pd.to_datetime(df_seasons['ds'])
        
        # Merge datasets
        df = pd.merge(df_summary, df_seasons, on='ds', how='left')
        
        # Feature engineering
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['ds'].dt.month
        df['day_of_month'] = df['ds'].dt.day
        
        # One-hot encode seasons
        seasons_dummy = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, seasons_dummy], axis=1)
        
        # Create lag features
        for lag in [1, 7, 14]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            df[f'orders_lag_{lag}'] = df['num_orders'].shift(lag)
        
        # Drop rows with NaN values from lag features
        df = df.dropna()
        
        return df
    
    def create_sequences(self):
        features_list = []
        sales_targets = []
        orders_targets = []
        
        numeric_features = ['sales', 'num_orders', 'avg_order_value', 'day_of_week', 
                           'is_weekend', 'month', 'day_of_month'] + \
                          [col for col in self.df.columns if 'season_' in col or 'lag_' in col]
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[numeric_features])
        
        for i in range(len(self.df) - self.seq_len - self.horizon):
            # Input sequence
            seq_features = scaled_features[i:i + self.seq_len]
            
            # Targets (sales and orders for horizon period)
            sales_target = self.df['sales'].iloc[i + self.seq_len:i + self.seq_len + self.horizon].values
            orders_target = self.df['num_orders'].iloc[i + self.seq_len:i + self.seq_len + self.horizon].values
            
            features_list.append(seq_features)
            sales_targets.append(sales_target)
            orders_targets.append(orders_target)
        
        return (torch.FloatTensor(features_list), 
                torch.FloatTensor(sales_targets), 
                torch.FloatTensor(orders_targets))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets_sales[idx], self.targets_orders[idx]

# 2. Time Series Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, horizon=7, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-head output for sales and orders
        self.sales_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon)
        )
        
        self.orders_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer processing
        encoded = self.transformer(x)
        
        # Use the last time step for forecasting
        last_encoded = encoded[:, -1, :]
        
        # Generate forecasts for both targets
        sales_pred = self.sales_head(last_encoded)
        orders_pred = self.orders_head(last_encoded)
        
        return sales_pred, orders_pred

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 3. Training Function
def train_model(model, train_loader, val_loader, epochs=200, patience=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_sales, batch_orders in train_loader:
            optimizer.zero_grad()
            sales_pred, orders_pred = model(batch_features)
            
            sales_loss = nn.MSELoss()(sales_pred, batch_sales)
            orders_loss = nn.MSELoss()(orders_pred, batch_orders)
            loss = sales_loss + orders_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_sales, batch_orders in val_loader:
                sales_pred, orders_pred = model(batch_features)
                sales_loss = nn.MSELoss()(sales_pred, batch_sales)
                orders_loss = nn.MSELoss()(orders_pred, batch_orders)
                val_loss += (sales_loss + orders_loss).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 4. Prediction Function
def predict_future(model, dataset, future_days=7):
    model.eval()
    
    # Get the most recent sequence
    last_sequence = dataset.features[-1:].clone()
    
    predictions_sales = []
    predictions_orders = []
    
    with torch.no_grad():
        current_sequence = last_sequence
        
        for day in range(future_days):
            # Predict next day
            sales_pred, orders_pred = model(current_sequence)
            
            # Store predictions
            predictions_sales.append(sales_pred[0, 0].item())
            predictions_orders.append(orders_pred[0, 0].item())
            
            # Update sequence for next prediction (simplified - in practice you'd update features)
            if day < future_days - 1:
                # This is a simplified update - you'd want to properly update all features
                current_sequence = torch.roll(current_sequence, shifts=-1, dims=1)
                # Update the last position with new prediction (this is simplified)
                # In a real scenario, you'd update all relevant features
    
    return predictions_sales, predictions_orders

# 5. Main Execution
def main():
    # Initialize dataset
    dataset = RestaurantDataset(
        'date_summary.csv',
        'date_seasons.csv', 
        'date_itemquantity.csv',
        seq_len=30,
        horizon=7
    )
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_dim = dataset.features.shape[-1]  # Number of features
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        horizon=7
    )
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, epochs=200)
    
    # Make predictions
    print("\nMaking predictions...")
    future_sales, future_orders = predict_future(model, dataset, future_days=7)
    
    # Print results
    print("\nNext 7 days predictions:")
    for i, (sales, orders) in enumerate(zip(future_sales, future_orders)):
        print(f"Day {i+1}: Sales = {sales:.0f}, Orders = {orders:.0f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(future_sales, 'o-', label='Predicted Sales')
    plt.title('Future Sales Prediction')
    plt.xlabel('Days Ahead')
    plt.ylabel('Sales')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(future_orders, 'o-', label='Predicted Orders', color='orange')
    plt.title('Future Orders Prediction')
    plt.xlabel('Days Ahead')
    plt.ylabel('Number of Orders')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()