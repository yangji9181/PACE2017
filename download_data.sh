SOURCE='http://lbai5.web.engr.illinois.edu/data/'
mkdir data
mkdir data/gowalla
cd data
wget $SOURCE/inter_small.pkl
wget $SOURCE/testdata_small.pkl
wget $SOURCE/traindata_small.pkl
cd gowalla
wget $SOURCE/gowalla/visited_spots.txt
wget $SOURCE/gowalla/spot_category.txt
wget $SOURCE/gowalla/user_network.txt
wget $SOURCE/gowalla/visited_spots.txt
wget $SOURCE/gowalla/Readme.txt


