#!/usr/bin/env python

from __future__ import print_function,division
import numpy as np
import argparse
import os,subprocess,sys
from collections import defaultdict
from Bio import SeqIO,AlignIO

######################################################

class HMM:
   def __init__(self):
      self.indices = list()
      self.lookup = dict()

class Groups:
   def __init__(self):
      self.names = set()
      self.group_from_id = defaultdict()
      self.files = []
   def __len__(self):
      return(len(self.names))

######################################################

def main(args):

   # ./proust.py -a nad5/all_w_key.mfa -g nad5/P_ids.txt

   groups = get_groups(args.aln_file,args.groups_file)

   print(groups.names)

   sum_re = defaultdict(float)

   for target in groups.names:

      print('Target:',target)

      (aln_a_file,aln_b_file) = split_alignment(args.aln_file,target,groups)

      hmm_file_a = mk_hmm(aln_a_file,args.prior)
      hmm_file_b = mk_hmm(aln_b_file,args.prior)

      hmm_a = parse_hmm(hmm_file_a)
      hmm_b = parse_hmm(hmm_file_b)

      aln_obj_a = AlignIO.read(aln_a_file,'fasta')
      aln_obj_b = AlignIO.read(aln_b_file,'fasta')

      common_indices = sorted(set(hmm_b.indices).intersection(set(hmm_a.indices)),key=int)

      for idx in common_indices:
         probs_a = hmm_a.probs[hmm_a.lookup[idx]]
         probs_b = hmm_b.probs[hmm_b.lookup[idx]]
         real_probs_a = np.exp(-probs_a)
         real_probs_b = np.exp(-probs_b)
         rel_entropy = np.sum((real_probs_a * np.log(real_probs_a/real_probs_b)))
         # using log rules (tests ok):
         # rel_entropy = np.sum(real_probs_a * (-probs_a + probs_b))
         column = int(idx) - 1
         print("sf: {0:6s} re:{1:3d} {2:6.2f}".format(target,int(idx),rel_entropy),end='')
         print(' ',aln_obj_a[:,column],':',aln_obj_b[:,column])
         sum_re[idx] += rel_entropy

   sum_re_arr = []
   for idx in common_indices:
      sum_re_arr.append(sum_re[idx])
   np_sum_re = np.array(sum_re_arr,dtype=np.float32)
   mu = np.mean(np_sum_re)
   sd = np.std(np_sum_re)

   sites_over_cutoff = 0
   for idx in common_indices:
      column = int(idx) - 1
      z = (sum_re[idx] - mu) / sd
      if sum_re[idx] > args.re_cut:
#         print('rel_ent:',idx,sum_re[idx],'z:',z,aln_obj_a[:,column],':',aln_obj_b[:,column])
         print("re:{0:4d} {1:6.2f} z: {2:6.2f}".format(int(idx),sum_re[idx],z),end='')
         print(' ',aln_obj_a[:,column],':',aln_obj_b[:,column])
         sites_over_cutoff += 1

   simple_score = sites_over_cutoff / len(common_indices)
   print(args.aln_file,'SCORE:',simple_score*100)

   exit()

######################################################

def mk_hmm(aln_file,prior):

   hmm_file = aln_file + '_hmm'

   if prior == None:
      hmmbuild_com = 'hmmbuild -o /dev/null ' + hmm_file + ' ' + aln_file
   else:
      hmmbuild_com = 'hmmbuild --p' + prior + ' -o /dev/null ' + hmm_file + ' ' + aln_file

   print(hmmbuild_com)
   subprocess.call(hmmbuild_com,shell=True)

   return hmm_file

######################################################

def split_alignment(aln_file,target,groups):

   target_mfa_file = 'tmp_proust_mfa_' + target
   rest_mfa_file = 'tmp_proust_mfa_rest'

   target_fh = open(target_mfa_file,'w')
   rest_fh = open(rest_mfa_file,'w')

   for seq_rec in SeqIO.parse(aln_file,'fasta'):
      if seq_rec.id in groups.group_from_id:
         if groups.group_from_id[seq_rec.id] == target:
            print(seq_rec.format('fasta'),end='',file=target_fh)
         else:
            print(seq_rec.format('fasta'),end='',file=rest_fh)
      else:
         print('Missing from groups:',seq_rec.id,file=sys.stderr)
         exit()

   target_fh.close()
   rest_fh.close()

   return (target_mfa_file,rest_mfa_file)

######################################################

def get_groups(aln_file,groups_file):

   groups = Groups()

   seen = set()

   with open(groups_file) as fh:
      for line in fh:
         line = line.rstrip()
         parts = line.split()
         id = parts[0]
         try:
            group = parts[1]
         except:
            group = '1'
         print(id,group)
         groups.group_from_id[id] = group
         groups.names.add(group)

   # get the ids that aren't in the groups file

   for seq_rec in SeqIO.parse(aln_file,'fasta'):
      if seq_rec.id in seen:
         print(seq_rec.id,' seen more than once',file=sys.stderr)
      else:
         seen.add(seq_rec.id)
      if seq_rec.id not in groups.group_from_id:
         print(seq_rec.id,' not defined in groups file',file=sys.stderr)
         groups.group_from_id[seq_rec.id] = '0'
         groups.names.add('0')

   if len(groups) == 1:
      print('Only one group - exiting.',file=sys.stderr)
      exit()

   return groups

######################################################

def parse_hmm(hmm_file):

   model = HMM()

   with open(hmm_file) as fh:
      lines = fh.readlines()

   main_model = 0
   i = 0
   hmm_probs = list()
   indices = list()
   while i < len(lines):
      line = lines[i]
#      print(line,end='')
      line = line.rstrip()
      parts = line.split()
      if line == '//':
         break
      if parts[0] == 'HMM':
#         print(i)
         alphabet = parts[1:]
         alphabet_len = len(alphabet)
#         print(' '.join(alphabet))
         i += 1 # skip match state transitions
         tag = lines[i+1].split()[0]
         if tag == 'COMPO': # line is optional
            i += 1
         i += 2 # HMM initiation lines
         main_model = 1
#         print(i)
         i += 1
      if main_model:
#         print(i,lines[i],end='')
         prob_line = lines[i]
         parts = lines[i].split()
#         print(' '.join(parts[1:alphabet_len+1]))
         hmm_probs.append(parts[1:alphabet_len+1])
         model.indices.append(parts[alphabet_len+1])
         i += 2
      i += 1
#   print(hmm_probs)
   model.probs = np.array(hmm_probs,dtype=np.float32)

   for i,posn in enumerate(model.indices):
      model.lookup[posn] = i

#   print(model.indices)

   return model

#   np_hmm_probs = -np_hmm_probs
#   print(np.exp(np_hmm_probs).sum(axis=1))


######################################################

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description = '')
   parser.add_argument('-a','--aln',action='store',dest='aln_file',help='alignment file',required=True)
   parser.add_argument('-g','--groups',action='store',dest="groups_file",help='groups file',required=True)
   parser.add_argument('-r','--re',action='store',dest="re_cut",help='relative entropy cutoff',type=float)
   parser.add_argument('-p','--prior',action='store',dest="prior",help='hmmbuild prior, default Dirichlet')

   args = parser.parse_args()

   if args.re_cut == None:
      args.re_cut = 0.0

   main(args)

