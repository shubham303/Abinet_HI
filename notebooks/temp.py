#!/usr/bin/env python
# coding: utf-8

# # 82841986 is_char and is_digit

# # 82075350 regrex non-ascii and none-digit

# ## 86460763 left

# In[ ]:


import os
import random
import re
import pandas as pd

# In[ ]:


max_length = 25
min_length = 1
root = '../data'
charset = 'ऀँंःअआइईउऊऋऌऍएऐऑओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऺऻऽािीुूृॄॅेैॉोौ्ॎॐ॒॑॓॔ॖॗॠॡॢॣ।॥ॲ'
digits = '०१२३४५६७८९%/?:,.-'


# In[ ]:


def is_char(text, ratio=0.5):
	text = text.lower()
	length = max(len(text), 1)
	char_num = sum([t in charset for t in text])
	if char_num < min_length: return False
	if char_num / length < ratio: return False
	return True


def is_digit(text, ratio=0.5):
	length = max(len(text), 1)
	digit_num = sum([t in digits for t in text])
	if digit_num / length < ratio: return False
	return True


# # generate training dataset

# In[ ]:


with open('/home/shubham/Documents/MTP/text-recognition-models/ABINet/data/HI_train.txt', 'r') as file:
	lines = file.readlines()

# In[5]:


inp, gt = [], []
for line in lines:
	token = line.lower().split()
	for text in token:
		#text = re.sub('[^0-9a-zA-Z]+', '', text)
		if len(text) < min_length:
			# print('short-text', text)
			continue
		if len(text) > max_length:
			# print('long-text', text)
			continue
		inp.append(text)
		gt.append(text)

# In[6]:


train_voc = os.path.join(root, 'WikiText-103.csv')
pd.DataFrame({'inp': inp, 'gt': gt}).to_csv(train_voc, index=None, sep=' ')

# In[7]:


len(inp)

# In[8]:


inp[:100]


# # generate evaluation dataset

# In[9]:


def disturb(word, degree, p=0.3):
	if len(word) // 2 < degree: return word
	if is_digit(word): return word
	if random.random() < p:
		return word
	else:
		index = list(range(len(word)))
		random.shuffle(index)
		index = index[:degree]
		new_word = []
		for i in range(len(word)):
			if i not in index:
				new_word.append(word[i])
				continue
			if (word[i] not in charset) and (word[i] not in digits):
				# special token
				new_word.append(word[i])
				continue
			op = random.random()
			if op < 0.1:  # add
				new_word.append(random.choice(charset))
				new_word.append(word[i])
			elif op < 0.2:
				continue  # remove
			else:
				new_word.append(random.choice(charset))  # replace
		return ''.join(new_word)


# In[10]:


lines = inp
degree = 1
keep_num = 50000

random.shuffle(lines)
part_lines = lines[:keep_num]
inp, gt = [], []

for w in part_lines:
	w = w.strip().lower()
	new_w = disturb(w, degree)
	inp.append(new_w)
	gt.append(w)

eval_voc = os.path.join(root, f'WikiText-103_eval_d{degree}.csv')
pd.DataFrame({'inp': inp, 'gt': gt}).to_csv(eval_voc, index=None, sep=' ')

# In[11]:


list(zip(inp, gt))[:100]

# In[ ]:



