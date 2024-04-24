#!/bin/bash


echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks


echo ""
echo ""
echo ""
echo "****************** RTS50 Network ******************"
gdown https://drive.google.com/uc\?id\=1MyrVu4Liz5aNdm_0j7vCg_Ya-iRYgmUb -O pytracking/networks/rts50.pth
gdown https://drive.google.com/uc\?id\=19ElXtYFLxN4pIP0n7DgQtAYewsyVwMN- -O pytracking/networks/sta.pth.tar

# echo ""
# echo ""
# echo ""
# echo "****************** DiMP50 Network ******************"
# gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth
# gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth

# echo ""
# echo ""
# echo "****************** ATOM Network ******************"
# gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

# echo ""
# echo ""
# echo "****************** ECO Network ******************"
# gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth

echo ""
echo ""
echo "****************** Setting up environment ******************"
python3 -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python3 -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

echo ""
echo ""
echo "****************** Cloning evutils Repository ******************"
git clone --recurse-submodules git@git.ee.ethz.ch:pbl/research/event-camera/evutils.git
cd evutils
pip install -e .
cd ..
echo ""
echo ""
echo "****************** Installation complete! ******************"

echo ""
echo ""
echo "****************** More networks can be downloaded from the google drive folder https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O ******************"
echo "****************** Or, visit the model zoo at https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md ******************"
