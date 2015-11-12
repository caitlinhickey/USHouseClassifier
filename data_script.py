from csv import DictReader, DictWriter
import re

votes = list(DictReader(open("tgraves13thcongress.csv", 'r')))
bills = list(DictReader(open("bills93-113.csv", 'r')))

labels = {}
for line in votes:
	BoolVote = None
	if line['Indiv.Vote'] == 'Yea':
		BoolVote = True
	if line['Indiv.Vote'] == 'Nay':
		BoolVote = False
	if BoolVote == None:
		continue

	BillNum = re.search(r"\d+", line['Bill No.'])

	if BillNum:
		labels.update({BillNum.group():BoolVote})

text = {}
for line in bills:
	if line['Cong'] == "113":
		text.update({line['BillNum']:line['Title']})

# Save combined dictionary as CSV


o = DictWriter(open("train.csv", 'w'), ["No.", "Label", "Text"])
o.writeheader()
for BillNum in labels.keys():
	if text[BillNum]:
		d = {'No.':BillNum , 'Label':labels[BillNum] , 'Text':text[BillNum] }
		o.writerow(d)