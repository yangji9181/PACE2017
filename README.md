## Implementation of *PACE*, KDD 2017.

Please cite the following work if you find the code useful.

```
@inproceedings{yang2017bridging,
	Author = {Yang, Carl and Bai, Lanxiao and Zhang, Chao and Yuan, Quan and Han, Jiawei},
	Booktitle = {KDD},
	Title = {Bridging collaborative filtering and semi-supervised learning: a neural approach for poi recommendation},
	Year = {2017}
}
```

Contact: Carl Yang (yangji9181@gmail.com)

## Usage:
To run the code, you need to have Python3 and iPython Notebook installed.

* Visit `https://snap.stanford.edu/data/loc-gowalla.html` or `https://www.yelp.com/dataset/challenge` to download the Gowalla or Yelp datasets. Please refer to `dataset.py` the paper for data preprocessing.
* Start iPython Notebook Server `ipython3 notebook` and sequentially run cells in `train.ipynb`

If you are using remote machine, you can:
* Start iPython Notebook Server on remote machine: `ipython notebook --no-browser --port=8889`
* Redirect ssh connection to localhost `ssh -N -f -L localhost:8880:localhost:8889 <user>@<host>`
* Open browser and go to `<user>@<host>:8880`
