#import re
#tweet = "alpha dog is a artificial @dog 11:32days"
#datestr = "23-10-2002\n23/10/2002\n23 Oct 2015\n23/10/02 9 10 18\n2-Octobor-1045"
#find = [item for item in tweet.split(' ') if re.search('@\w{3}', item)]
#find2 = [item for item in tweet.split(' ') if re.findall(r'[dog]', item)]
#find3 = re.findall(r'[dog]',tweet)
#find4 = re.findall(r'\d{1,2}[-/\s][A-Za-z0-9]+[-/\s]\d{2,4}', datestr)
#print(find)
#print(find2)
#print(find3)
#print(find4)
#print(tweet.split(' '))
#find5 = re.findall(r'\d?\d:\d\dday\b',tweet)
#print(find5)
#print('abc'.groups())

import nltk
a = ['Cameron','cameron']
b = nltk.word_tokenize(a)
print(b)
