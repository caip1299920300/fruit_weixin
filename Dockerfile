FROM spellrun/tensorflow2-base
MAINTAINER wcp<1299920300@qq.com>

COPY readme.txt /usr/local/readme.txt

ADD apache-tomcat-9.0.54.tar.gz /usr/local/
ADD jdk-8u291-linux-x64.tar.gz /usr/local/

# 把当前的路径下的文件添加到镜像的code文件中
ADD . /tf2_project
# 设置code文件为当前目录
WORKDIR /tf2_project
# 下载requirements.txt中的flask和pymysql文件
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# 运行app.py脚本文件

EXPOSE 8080

CMD ["python", "weixin_predict.py"]