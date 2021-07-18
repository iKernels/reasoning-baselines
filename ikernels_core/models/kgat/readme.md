This folder contains a wrapper for the original KGAT models for fact checking described in *Liu, Z., Xiong, C., & Sun, M. (2020). [Kernel Graph Attention Network for Fact Verification.](https://doi.org/10.18653/v1/2020.acl-main.655) In ACL.*

The original source code of the (Liu et al., 2020) pipeline is available at [https://github.com/thunlp/KernelGAT](https://github.com/thunlp/KernelGAT).

Here, we wrapped portions of code from [https://github.com/thunlp/KernelGAT/blob/master/kgat/models.py](https://github.com/thunlp/KernelGAT/blob/master/kgat/models.py) into the AllenNLP `Model` interface.

KGAT code is distributed under the MIT license:
```
MIT License

Copyright (c) 2019 THUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
