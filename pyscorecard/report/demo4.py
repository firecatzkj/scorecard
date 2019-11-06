# -*- coding: utf8 -*-
from pyscorecard.report.base import MyRender




class Demo4(MyRender):
    pass

import pysnooper


@pysnooper.snoop()
def a(c,v):
    return c+v


if __name__ == '__main__':
    a(1,2)