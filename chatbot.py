# Importando as bibliotecas
import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2



########## PART 1 - PROCESSAMENTO DOS DADOS ##########



# Importando dataset
metadata, idx_q, idx_a = data_preprocessing.load_data(PATH = './')

# Dividindo o conjunto de dados no conjunto de Treinamento e no conjunto de Teste
(trainX, trainY), (testX, testY), (validX, validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()



########## PART 2 - CONSTRUINDO O MODELO SEQ2SEQ ##########



# construindo modelo seq2seq
model = seq2seq_wrapper.Seq2Seq(xseq_len = xseq_len,
                                yseq_len = yseq_len,
                                xvocab_size = xvocab_size,
                                yvocab_size = yvocab_size,
                                ckpt_path = './weights',
                                emb_dim = emb_dim,
                                num_layers = 3)



########## PART 3 - TREINANDO O MODELO SEQ2SEQ ##########



# Veja o treinamento em seq2seq_wrapper.py



########## PART 4 - TESTANDO O MODELO SEQ2SEQ ##########



# Carregando os Pesos e Executando a Sessão
session = model.restore_last_session()

# obtendo a resposta prevista do ChatBot
def respond(question):
    encoded_question = data_utils_2.encode(question, w2idx, limit['maxq'])
    answer = model.predict(session, encoded_question)[0]
    return data_utils_2.decode(answer, idx2w) 

# Configurando o bate-papo
while True :
  question = input("You: ")
  answer = respond(question)
  print ("ChatBot: "+answer)
