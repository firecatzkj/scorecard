# -*- coding: utf8 -*-
import pandas as pd
import pyecharts11.options as opts
from pyecharts11.charts import Bar
from jinja2 import Markup, Environment, FileSystemLoader
from pyecharts11.globals import CurrentConfig
from pyecharts11.render.engine import RenderEngine
from pyecharts11.components import Table
from pyecharts11.commons.utils import write_utf8_html_file

# 关于 CurrentConfig，可参考 [基本使用-全局变量]
CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("templates"))
CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"


def bar_base() -> Bar:
    c = (
        Bar()
        .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
        .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
        .add_yaxis("商家B", [15, 25, 16, 55, 48, 8])
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="我是副标题"))
    )
    return c


def table_base(df: pd.DataFrame):
    table = Table()

    # headers = ["City name", "Area", "Population", "Annual Rainfall"]
    # rows = [
    #     ["Brisbane", 5905, 1857594, 1146.4],
    #     ["Adelaide", 1295, 1158259, 600.5],
    #     ["Darwin", 112, 120900, 1714.7],
    #     ["Hobart", 1357, 205556, 619.5],
    #     ["Sydney", 2058, 4336374, 1214.8],
    #     ["Melbourne", 1566, 3806092, 646.9],
    #     ["Perth", 5386, 1554769, 869.4],
    # ]
    headers = list(df.columns)
    rows = [list(df.iloc[i]) for i in range(len(df))]
    table.add(headers, rows)
    return table



bb = bar_base()
cc = bar_base()
dd = table_base(pd.DataFrame({
    "A1": [1,2,3,4],
    "A2": [1,2,3,4],
    "A3": [1,2,3,4],
    "A4": [1,2,3,4],
    "A5": [1,2,3,4]
}))
print(bb.get_options()["title"].opts[0]["text"])


bb._prepare_render()
cc._prepare_render()

reg = RenderEngine()
tmp = reg.env.get_template("mydemo.html")
html = tmp.render(c1=reg.generate_js_link(bb), c2=reg.generate_js_link(cc), c3=dd)
write_utf8_html_file("333.html", reg._replace_html(html))




# xx = bar_base().render("222.html", template_name="mydemo.html")
# print(xx)

# print(bar_base().get_options())
# print(bar_base().js_host)

# from pyecharts.charts import Page
# pp = Page()
# pp.render()

