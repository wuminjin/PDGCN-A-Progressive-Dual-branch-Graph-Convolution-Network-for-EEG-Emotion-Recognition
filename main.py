import os
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from datapipe import build_dataset, get_dataset
from PDGCN import PDGCN
from utils import set_seed, reset_seed

seed = 520
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_log(Network):
    version = 1
    print('***' * 20)
    while (1):  # 生成唯一日志文件
        dfile = './result/{}_{:.0f}.csv'.format(Network.__name__, version)
        if not os.path.exists(dfile):
            break
        version += 1
    print(dfile)
    df = pd.DataFrame()  # 创建一个空的pandas DataFrame对象
    df.to_csv(dfile)  # 将DataFrame保存到CSV文件dfile

    return dfile


def train(model, train_loader, crit, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        #Multiple Classes classification Loss function
        label = torch.argmax(data.y.view(-1,classes), axis=1)
        label = label.to(device)#, dtype=torch.long) #, dtype=torch.int64)

        output, _ = model(data.x, data.edge_index, data.batch)

        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step() 

    return loss_all / len(train_dataset)

def evaluate(model, loader, save_result=False):
    model.eval()

    predictions = []
    labels = [] 

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1,classes)
            data = data.to(device)
            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy() 
            
            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    #AUC score estimation 
    AUC = roc_auc_score(labels, predictions, average='macro')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    #Accuracy 
    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    acc = accuracy_score(labels, predictions)

    return AUC, acc, f1


if __name__ == "__main__":

    subjects = 15
    epochs = 200
    classes = 3  # Num. of classes
    Network = DSGCN

    #1、创建日志
    dfile = create_log(Network)

    #d、若数据集未创建，创建数据集
    build_dataset(subjects)
    print('Cross Validation')

    #3、进行训练
    result_data = []
    all_last_acc = []
    all_last_AUC = []
    for cv_n in range(0, subjects):
        train_dataset, test_dataset = get_dataset(subjects, cv_n)

        # train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.d, random_state=seed4)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=seed)

        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)

        train_loader = DataLoader(train_dataset, batch_size=16, drop_last=False, shuffle=True,
                                  worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=16, worker_init_fn=worker_init_fn)
        reset_seed(seed)

        model = Network().to(device)

        # lr = 0.000005
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.0001)
        crit = torch.nn.CrossEntropyLoss() #

        epoch_data = []
        val_AUC_f, val_acc_f, val_f1_f = 0, 0, 0

        for epoch in range(epochs):
            t0 = time.time()
            loss = train(model, train_loader, crit, optimizer)
            train_AUC, train_acc, train_f1 = evaluate(model, train_loader)
            val_AUC, val_acc, val_f1 = evaluate(model, val_loader)
            test_AUC, test_acc, test_f1 = evaluate(model, test_loader)

            epoch_data.append([str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc])
            t1 = time.time()
            # print('V{:01d}, EP{:03d}, Loss:{:.5f}, AUC:{:.2f}, Acc:{:.2f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
            #           format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, (t1-t0)))
            print(
                f'V{cv_n:01d}, EP{epoch + 1:03d}, Loss:{loss:.5f}, AUC:{train_AUC:.2f}, Acc:{train_acc:.2f}, VAUC:{val_AUC:.2f}, Vacc:{val_acc:.2f}, TAUC: {test_AUC:.4f}, TAcc: {test_acc:.4f}, Time: {(t1 - t0):.2f}')

            if val_acc >= val_acc_f:
                val_AUC_f, val_acc_f, val_f1_f = val_AUC , val_acc, val_f1
                best_model_path = os.path.join("best_model/1", f'{Network.__name__}_sub{cv_n}.pth')
                torch.save(model.state_dict(), best_model_path)

        # 最终测试
        if best_model_path:
            # 加载验证集上性能最佳的模型
            # model.load_state_dict(torch.load(best_model_path))
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            test_AUC, test_acc, test_f1 = evaluate(model, test_loader)
            print('Final Test Results:')
            print(f'Test AUC: {test_AUC:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

        # print('Results::::::::::::')
        # print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.7f}, Acc:{:.7f}, VAUC:{:.2f}, Vacc:{:.2f}, Time: {:.2f}'.
        #               format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC_f, val_acc_f, (t1-t0)))

        result_data.append([str(cv_n), epoch+1, loss, test_AUC, test_acc, test_f1])

        df = pd.DataFrame(data=result_data, columns=['Fold', 'Epoch', 'Loss', 'test_AUC', 'test_acc', 'test_f1'])

        df.to_csv(dfile)


    df = pd.read_csv(dfile)
    output_file = "training_results.xlsx"
    df.to_excel(output_file, index=False, engine="openpyxl")

    lastacc = ['test_acc', df['test_acc'].mean()]
    lastauc = ['test_AUC', df['test_AUC'].mean()]
    print(lastacc)
    print(lastauc)
    print('*****************')


