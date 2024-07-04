import pickle

directory_path_training = './training/'

def SalvarQTable(q_table, filename='QTable.pkl'):
    with open(directory_path_training+filename, 'wb') as f:
        pickle.dump(q_table, f)

def CarregarQTable(filename='QTable.pkl'):
    with open(directory_path_training+filename, 'rb') as f:
        return pickle.load(f)

def SalvarAmbiente(max_qtd_supply,filename="QAmbiente.pkl"):
    with open(directory_path_training+filename, 'wb') as f:
        pickle.dump(max_qtd_supply, f)

def CarregarAmbiente(filename='QAmbiente.pkl'):
    with open(directory_path_training+filename, 'rb') as f:
        return pickle.load(f)
