import argparse
import time
import torch
import torch.nn.utils.prune as prune
from torch.utils.tensorboard import SummaryWriter
import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os
import copy
import data
import model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

parser = argparse.ArgumentParser(description='PyTorch SGNS and LogitSGNS Models',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='sgns',
                    help='model to use: sgns=SGNS, lsgns=LogitSGNS')
parser.add_argument('--data', type=str, default='data/text8',
                    help='location of the data corpus')
parser.add_argument('--valid', type=str, default=None,
                    help='location of the validation set')
parser.add_argument('--save_dir', type=str, default='embeddings',
                    help='path to save the word vectors')
parser.add_argument('--save_file', type=str, default='sgns',
                    help='path to save the word vectors')                    
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='batch size')
parser.add_argument('--window_size', type=int, default=5,
                    help='context window size')
parser.add_argument('--neg_num', type=int, default=5,
                    help='negative samples per training example')
parser.add_argument('--min_count', type=int, default=5,
                    help='number of word occurrences for it to be included in the vocabulary')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='epsilon to be used in the LogitSGNS model')
parser.add_argument('--gpu', default='0',
                    help='GPU to use')
parser.add_argument('--save_initial', type=bool,default=False,
                    help='save initial weights or not')

parser.add_argument('--pruned', type=bool,default=False,
                    help='train pruned model')  

parser.add_argument('--prune_amount', type=float, default=0.00,
                    help='percent of pruned weights')

parser.add_argument('--exp2', type=bool,default=False,
                    help='exp2 train only v_embeddings')  

parser.add_argument('--exp2_file',  type=str, default=None,
                    help='exp2 checkpoint')  

args = parser.parse_args()


if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

my_data = data.DataReader(args.data, args.min_count)
dataset = data.Word2vecDataset(my_data, args.window_size, args.neg_num)
dataloader = torch.utils.data.DataLoader(
  dataset, batch_size=args.batch_size, collate_fn=dataset.collate)

if args.valid != None:
  valid_dataset = data.ValidDataset(my_data, args.valid, args.window_size, args.neg_num)
  valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, batch_size=args.batch_size, collate_fn=valid_dataset.collate)

vocab_size = len(my_data.word2id)
if args.model == 'sgns': 
  skip_gram_model = model.SkipGramModel(vocab_size, args.emsize)
elif args.model == 'lsgns': 
  skip_gram_model = model.LogitSGNSModel(vocab_size, args.emsize, args.epsilon)
else: 
  print("No such model:", args.model)
  exit(1)


mask = None
if args.pruned:
  init_checkpoint = torch.load('epoch1_state_dict_sgns.pth')
  final_checkpoint = torch.load('final_state_dict_sgns.pth')
  skip_gram_model.load_state_dict(final_checkpoint)
  prune.l1_unstructured(skip_gram_model.v_embeddings, name='weight', amount=args.prune_amount)
  mask = skip_gram_model.v_embeddings.weight_mask
  mask=mask.bool()
  prune.remove(skip_gram_model.v_embeddings, 'weight')

  skip_gram_model.load_state_dict(init_checkpoint)

if args.exp2:
  checkpoint = torch.load(args.exp2_file)
  skip_gram_model.load_state_dict(checkpoint)
  skip_gram_model.u_embeddings.weight.requires_grad = False


# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda: skip_gram_model.cuda()

epoch_size = dataset.data_len // args.batch_size
optimizer = torch.optim.Adam(skip_gram_model.parameters())

if args.exp2:
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, skip_gram_model.parameters()))



writer = SummaryWriter('runs/' + args.save_file)




for epoch in range(args.epochs):
  last_time = time.time()
  last_words = 0

  total_loss = 0.0
  
  for step, batch in enumerate(dataloader):
    pos_u = batch[0].to(device)
    pos_v = batch[1].to(device)
    neg_v = batch[2].to(device)

    optimizer.zero_grad()
    loss = skip_gram_model.forward(pos_u, pos_v, neg_v)
    loss.backward()

    if args.pruned:
      skip_gram_model.v_embeddings.weight.grad.data[~mask]=0



    optimizer.step()

    total_loss += loss.item()

    if step % (epoch_size // 10) == 10:
      print('%.2f' % (step * 1.0 / epoch_size), end=' ')
      print('loss = %.3f' % (total_loss / (step + 1)), end=', ')
      now_time = time.time()
      now_words = step * args.batch_size
      wps = (now_words - last_words) / (now_time - last_time)
      print('wps = ' + str(int(wps)))
      last_time = now_time
      last_words = now_words
  
  if args.save_initial:
    if epoch==0:
      torch.save(skip_gram_model.state_dict(), f"epoch1_state_dict_sgns.pth")

  print("Epoch: " + str(epoch + 1), end=", ")
  print("Loss = %.3f" % (total_loss / epoch_size), end=", ")
  writer.add_scalar('Loss/train', (total_loss / epoch_size), epoch)

  # Compute validation loss
  if args.valid != None:
    valid_epoch_size = valid_dataset.data_len // args.batch_size
    valid_total_loss = 0.0

    for valid_step, valid_batch in enumerate(valid_dataloader):
      pos_u = valid_batch[0].to(device)
      pos_v = valid_batch[1].to(device)
      neg_v = valid_batch[2].to(device)

      with torch.no_grad():
        valid_loss = skip_gram_model.forward(pos_u, pos_v, neg_v)

      valid_total_loss += valid_loss.item()

    print("Valid Loss = %.3f" % (valid_total_loss / valid_epoch_size), end=', ')
    writer.add_scalar('Loss/val', (valid_total_loss / valid_epoch_size), epoch)

  torch.save(skip_gram_model.state_dict(), os.path.join(args.save_dir, args.save_file+".pth"))
  skip_gram_model.save_embedding(my_data.id2word, os.path.join(args.save_dir, args.save_file))
  wv_from_text = KeyedVectors.load_word2vec_format(os.path.join(args.save_dir, args.save_file), binary=False)
  ws353 = wv_from_text.evaluate_word_pairs(os.path.join('testsets','wordsim353.tsv'))
  google = wv_from_text.evaluate_word_analogies(os.path.join('testsets','questions-words.txt'))
  print('WS353 = %.3f' % ws353[0][0], end=', ')
  print('Google = %.3f' % google[0])
  writer.add_scalar('WS353', ws353[0][0], epoch)
  writer.add_scalar('Google', google[0], epoch)