#Run with Python3/GPU-enabled runtime

#Train/Valid/Test set for predicting median gene expression levels in the human
wget -r -np -nH --reject "index.html*" --cut-dirs 6 \
 https://krishna.gs.washington.edu/content/members/vagar/Xpresso/data/datasets/pM10Kb_1KTest/
#Train/Valid/Test set for predicting median gene expression levels in the mouse
wget -r -np -nH --reject "index.html*" --cut-dirs 6 \
 https://krishna.gs.washington.edu/content/members/vagar/Xpresso/data/datasets/pM10Kb_1KTest_Mouse/
#Prepare set of input sequences to generate predictions
wget https://xpresso.gs.washington.edu/data/Xpresso-predict.zip
unzip Xpresso-predict.zip

#set up dependencies
pip install --user biopython
pip install --user hyperopt

