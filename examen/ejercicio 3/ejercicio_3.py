import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import pickle


class MyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))

    return mini_batches


def add_dummies(df, col_name, prefix=None, drop_first=True):
    dummies = pd.get_dummies(df[col_name], drop_first=drop_first, prefix=prefix)
    # Borro la columna y agrego los dummies de esta al dataset
    df = df.drop(col_name, axis=1)
    df = df.join(dummies)
    return df


def one_hot(df, columns_dict):
    for col in columns_dict:
        df = add_dummies(df, col, prefix=columns_dict[col])
    return df


class NNet(torch.nn.Module):
    def __init__(self, n_features, n_extra_layers, layer_neurons=100):
        super().__init__()
        self.layers = list()
        linear = torch.nn.Linear(in_features=n_features, out_features=layer_neurons, bias=True)
        self.layers.append(linear)
        relu = torch.nn.ReLU()
        self.layers.append(relu)
        for _ in range(n_extra_layers):
            linear = torch.nn.Linear(in_features=layer_neurons, out_features=layer_neurons, bias=True)
            self.layers.append(linear)
            relu = torch.nn.ReLU()
            self.layers.append(relu)
        self.output = torch.nn.Linear(in_features=layer_neurons, out_features=1, bias=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


################## OPEN DATASET ##################
dataset_path = "../dataset_2.csv"
dataset = pd.read_csv(dataset_path)

################## LABEL ##################
dataset["purchase_label"] = dataset.Purchase.apply(lambda label: 0 if label < 9000 else 1)
dataset = dataset.drop('Purchase', axis=1)

################## CREATING DIFFERENT DATASETS FOR TESTING ##################

datasets = list()
################## OHE sobre todas las cols ##################
dataset['Product_Category_2'].fillna(0, inplace=True)
dataset['Product_Category_3'].fillna(0, inplace=True)

# ONE HOT ENCODING
one_hot_columns_dict = {
    "Age": "age",
    "Gender": "gender",
    "Occupation": "occ",
    "City_Category": "city",
    "Stay_In_Current_City_Years": "stay",
    "Product_Category_1": "prod_cat_1",
    "Product_Category_2": "prod_cat_2",
    "Product_Category_3": "prod_cat_3"
}

dataset_1 = one_hot(dataset, one_hot_columns_dict)
datasets.append(("dataset_1", dataset_1))

################## OHE sobre algunas cols ##################
# ONE HOT ENCODING
one_hot_columns_dict = {
    "Age": "age",
    "Gender": "gender",
    "City_Category": "city",
    "Stay_In_Current_City_Years": "stay",
}

dataset_2 = one_hot(dataset, one_hot_columns_dict)
datasets.append(("dataset_2", dataset_2))

################## SIN Product_Category_* y OHE sobre algunas cols ##################
# ONE HOT ENCODING
one_hot_columns_dict = {
    "Age": "age",
    "Gender": "gender",
    "City_Category": "city",
    "Stay_In_Current_City_Years": "stay",
}

dataset_3 = one_hot(dataset, one_hot_columns_dict)
dataset_3 = dataset_3.drop(columns=["Product_Category_1", "Product_Category_2", "Product_Category_3"])
datasets.append(("dataset_3", dataset_3))

################## SIN Product_Category 2 y 3 y OHE sobre algunas cols ##################
# ONE HOT ENCODING
one_hot_columns_dict = {
    "Age": "age",
    "Gender": "gender",
    "City_Category": "city",
    "Stay_In_Current_City_Years": "stay",
}

dataset_4 = one_hot(dataset, one_hot_columns_dict)
dataset_4 = dataset_4.drop(columns=["Product_Category_2", "Product_Category_3"])
datasets.append(("dataset_4", dataset_4))

################## SIN Occupation y OHE sobre algunas cols ##################
# ONE HOT ENCODING
one_hot_columns_dict = {
    "Age": "age",
    "Gender": "gender",
    "City_Category": "city",
    "Stay_In_Current_City_Years": "stay",
}

dataset_5 = one_hot(dataset, one_hot_columns_dict)
dataset_5 = dataset_5.drop(columns=["Occupation"])
datasets.append(("dataset_5", dataset_5))

################## TRAINING ##################
n_epochs = 1000
for ds_name, ds in datasets:
    X = ds.drop(columns=["purchase_label", 'Product_ID', 'User_ID'])
    y = ds["purchase_label"]

    ################## NORMALIZE ##################
    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    ################## DATALOADERS ##################
    X_train, X_valid, y_train, y_valid = train_test_split(X_norm, y, test_size=0.25, random_state=12)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_valid = X_valid.to_numpy()
    y_valid = y_valid.to_numpy()

    y_train = y_train.reshape(-1)
    y_valid = y_valid.reshape(-1)

    train = MyDataset(X_train, y_train)
    valid = MyDataset(X_valid, y_valid)
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size=64, shuffle=True)
    ################## TRAINING ##################
    n_features = X_norm.shape[1]

    nnets = [("0_extra_100_neurons", NNet(n_features, n_extra_layers=0, layer_neurons=100)),
             ("1_extra_100_neurons", NNet(n_features, n_extra_layers=1, layer_neurons=100)),
             ("2_extra_100_neurons", NNet(n_features, n_extra_layers=2, layer_neurons=100)),
             ("3_extra_100_neurons", NNet(n_features, n_extra_layers=3, layer_neurons=100))]

    device = ""
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(device)

    for nnet_name, nnet in nnets:
        optimizer = torch.optim.Adam(nnet.parameters(), lr=0.001)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum")

        print("*************** {} {} ***************".format(ds_name, nnet_name))

        nnet.to(device)
        loss_list = list()
        valid_loss_list = list()
        train_auc = list()
        valid_auc = list()

        for epoch in range(n_epochs):  # epochs
            epoch_loss = 0
            epoch_valid_loss = 0
            epoch_acc = 0
            running_y_score = list()
            running_y_label = list()
            valid_y_score = list()
            valid_y_label = list()

            nnet.train()
            for i, data in enumerate(train_dataloader):  # batches
                # Datos del batch
                X_batch, y_batch = data
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                # forward
                optimizer.zero_grad()  # que el optim ponga los gradientes en 0
                y_batch_score = nnet(X_batch).reshape(-1)
                y_batch_hat = torch.sigmoid(y_batch_score).reshape(-1)

                # backpropagation
                loss = loss_function(y_batch_score, y_batch)
                loss.backward()  # hacer gradientes de score

                # actualizar parametros
                optimizer.step()

                epoch_loss += loss.item()
                running_y_score += list(y_batch_score.detach().cpu().numpy())
                running_y_label += list(y_batch.detach().cpu().numpy())

            nnet.eval()

            with torch.no_grad():
                for i, data in enumerate(valid_dataloader):
                    X_valid_batch, y_valid_batch = data
                    X_valid_batch = X_valid_batch.to(device).float()
                    y_valid_batch = y_valid_batch.to(device).float()

                    # forward
                    y_valid_batch_score = nnet(X_valid_batch).reshape(-1)
                    y_valid_batch_hat = torch.sigmoid(y_valid_batch_score).reshape(-1)

                    valid_loss = loss_function(y_valid_batch_score, y_valid_batch)

                    epoch_valid_loss += valid_loss.item()

                    valid_y_score += list(y_valid_batch_score.detach().cpu().numpy())
                    valid_y_label += list(y_valid_batch.detach().cpu().numpy())

            fpr, tpr, _ = metrics.roc_curve(running_y_label, running_y_score)
            epoch_auc = metrics.auc(fpr, tpr)

            val_fpr, val_tpr, _ = metrics.roc_curve(valid_y_label, valid_y_score)
            epoch_valid_auc = metrics.auc(val_fpr, val_tpr)

            train_auc.append(epoch_auc)
            valid_auc.append(epoch_valid_auc)
            loss_list.append(epoch_loss)
            valid_loss_list.append(epoch_valid_loss)

            print("Epoch:", epoch,
                  "\tTraining Loss:", epoch_loss,
                  "\tValidation Loss:", epoch_valid_loss,
                  "\tAUC:", epoch_auc)

            if epoch % 100 == 0:
                with open('output/loss_list_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
                    pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('output/valid_loss_list_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
                    pickle.dump(valid_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('output/train_auc_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
                    pickle.dump(train_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('output/valid_auc_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
                    pickle.dump(valid_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('output/nnet_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
                    pickle.dump(nnet, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('output/loss_list_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
            pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('output/valid_loss_list_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
            pickle.dump(valid_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('output/train_auc_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
            pickle.dump(train_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('output/valid_auc_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
            pickle.dump(valid_auc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('output/nnet_{}_{}.pickle'.format(ds_name, nnet_name), 'wb') as handle:
            pickle.dump(nnet, handle, protocol=pickle.HIGHEST_PROTOCOL)
