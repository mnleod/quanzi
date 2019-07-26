# Quanzi

This repository provides an implementation for computing guanxi circle as it is described in:
 [Chapter 4 Measurement of Guanxi Circles: Using Qualitative Study to Modify Quantitative Measurement: Interdisciplinary Approaches and Case Studies](https://books.google.com.hk/books?id=59WRDgAAQBAJ&pg=PA73&lpg=PA73&dq=The+Measurement+of+Guanxi+Circles%E2%80%94Using+Qualitative+Study+to+Modify+Quantitative+Measurement&source=bl&ots=WyyJm7Rv-x&sig=ACfU3U0WilrNA95u9yKqYHseFxwRiyDO2w&hl=en&sa=X&ved=2ahUKEwjLxc_nv9HjAhUJHKYKHbyPAikQ6AEwBHoECAgQAQ#v=onepage&q=The%20Measurement%20of%20Guanxi%20Circles%E2%80%94Using%20Qualitative%20Study%20to%20Modify%20Quantitative%20Measurement&f=false)

## Install
```bash
pip install git+https://github.com/yuezhang18/quanzi
```
or
```bash
git clone https://github.com/yuezhang18/quanzi.git
cd quanzi
python setup.py install
```


## Usage

use as module
```python
from quanzi import UndirectQuanzi
q = UndirectQuanzi()
q.read_csv("~/input.csv")
result = q.run()
```
use to generate output
```python
from quanzi import UndirectQuanzi
q = UndirectQuanzi()
q.read_csv("~/input.csv")
q.run()
q.to_csv("~/output.csv")
```