#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[105]:


df = pd.read_csv(
	"data/household_power_consumption.txt",
	sep=';',
	low_memory=False
)


# In[106]:


df.head()


# In[107]:


df.shape


# In[108]:


df.info()


# We can see that data has inappropriate format (Dtype). All the column except Date and Time should be float. Computer will not work with strings, we try to avoid this format.

# In[109]:


columns = list(df.columns)
columns.remove("Date")
columns.remove("Time")

for column in columns:
	df[column] = pd.to_numeric(df[column], errors="coerce")


# In[110]:


df.info()


# To get away from string columns, we combine Date and Time columns and put them as index

# In[111]:


df["full_time"] = pd.to_datetime(df["Date"] + ' ' + df["Time"], dayfirst=True)

df.set_index("full_time", inplace=True)

df.drop(columns=["Date", "Time"], inplace=True)


# In[112]:


df.head()


# In[113]:


df.describe()


# In[114]:


df.isna().sum(axis=0)


# In[115]:


nan_rows = df.isna().all(axis=1)

nan_rows.sum()


# In[135]:


nan_rows = nan_rows.resample('D').sum()

dates = nan_rows.index
values = nan_rows.values

plt.bar(dates, values)
plt.xticks(rotation=45)


# 

# We have to fill these NaN values. We can do it by replacing them with median or mean of the column. I think median is more appropriate, because electricity consumptions distribution is very skewed. Mean is very vulnurable to outliers or skewed distributions

# In[117]:


interval = 21
interval_means = df.rolling(window=interval, center=True, min_periods=int(interval / 2)).mean()

df = df.fillna(interval_means)


# In[136]:


df = df.dropna(axis=0)


# In[138]:


df.isnull().sum()


# In[137]:


df.hist(figsize=(15, 10), bins=40)


# from these graphs we can say that our data has a lot of outliers. Normal distribution has only Voltage feature.

# In[120]:


sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


# Standard train_test_split would be a bad option for us, because we work with sequential data. We must cut the data chronologically;

# In[143]:


print(df.index[0])
print(df.index[len(df) - 1])


# In[146]:


split_id = pd.to_datetime("2009-12-30 00:00:00", dayfirst=True)

X = df.drop(columns=["Global_active_power"])
y = df["Global_active_power"]

X_train = X.loc[:split_id, :]
X_test = X.loc[split_id:, :]

y_train = y.loc[:split_id]
y_test = y.loc[split_id:]


# In[122]:


print(X_train.shape)
print(y_train.shape)


# In[123]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


VAL_SPLIT = 0.1
val_split_id = int(len(X_train) * (1 - VAL_SPLIT))

X_tr = X_train[:val_split_id]
X_val = X_train[val_split_id:]

y_tr = y_train.values[:val_split_id]
y_val = y_train.values[val_split_id:]


# In[125]:


SEQ_LEN = 60
def create_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)


# An LSTM cannot learn from single time steps. It needs a sequence of past observations to predict a future value so we convert time-series into sliding windows

# In[126]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[127]:


X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test, y_test.values, SEQ_LEN)


# In[128]:


X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(1).to(device)

X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1).to(device)

X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(device)


# In[129]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[130]:


model = LSTM(
    input_size=X_tr_seq.shape[2],
    hidden_size=64,
    output_size=1,
    num_layers=2
).to(device)


# In[ ]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optimizer,
	mode="min",
	factor=0.25,
	patience=4
)


# In[132]:


isn


# As a loss function we use Mean Squared Error (MSE) because it is regression task. Also it penalizes larger errors more strongly.

# In[ ]:


EPOCHS = 10
BATCH_SIZE = 64

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    start_time = time.time()
    epoch_train_loss = 0
    permutation = torch.randperm(X_tr_t.size(0))

    for i in range(0, X_tr_t.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x = X_tr_t[indices]
        batch_y = y_tr_t[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # time left
        batches_done = min(i + BATCH_SIZE, X_tr_t.size(0))
        progress = batches_done / X_tr_t.size(0)
        elapsed = time.time() - start_time
        eta = elapsed * (1 - progress) / progress if progress > 0 else 0
        print(f"\rEpoch {epoch+1}/{EPOCHS} - Batch {batches_done}/{X_tr_t.size(0)} "
              f"- Train Loss: {loss.item():.4f} - time left: {eta:.1f}s", end="")

    epoch_train_loss /= len(X_tr_t)
    train_losses.append(epoch_train_loss)

    # batch validation due to bad gpu
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for i in range(0, X_val_t.size(0), BATCH_SIZE):
            batch_x = X_val_t[i:i+BATCH_SIZE]
            batch_y = y_val_t[i:i+BATCH_SIZE]
            preds = model(batch_x)
            batch_loss = criterion(preds, batch_y)
            val_loss_total += batch_loss.item() * batch_x.size(0)

    val_loss = val_loss_total / X_val_t.size(0)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    print(f"\rEpoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {val_loss:.4f}")


# In[ ]:


plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


# In[ ]:


model.eval()

y_preds = []

BATCH_SIZE_EVAL = 512

with torch.no_grad():
    for i in range(0, X_test_t.size(0), BATCH_SIZE_EVAL):
        batch_x = X_test_t[i:i+BATCH_SIZE_EVAL]
        preds = model(batch_x)
        y_preds.append(preds.cpu())

y_pred = torch.cat(y_preds).numpy()


# In[ ]:


rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
mae = mean_absolute_error(y_test_seq, y_pred)
r2 = r2_score(y_test_seq, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")


# The coefficient of determination (R²) measures how well the regression model explains the variability of the target variable. It represents the proportion of variance in the true values that is captured by the model’s predictions. An R² value of **0.95** indicates that approximately **95.5% of the variability** in global active power consumption is explained by the LSTM model. This suggests that the model successfully captures the underlying temporal patterns and relationships present in the data. R² is particularly useful in regression tasks because it provides a scale-independent measure of model performance and allows direct comparison with baseline models, such as predicting the mean value.

# The high R² score confirms that the proposed LSTM model generalizes well and explains most of the variance in household power consumption.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.plot(y_test_seq[:1000], label="True")
plt.plot(y_pred[:1000], label="Predicted")
plt.legend()
plt.title("Global Active Power Prediction")
plt.show()

