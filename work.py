def all():
    import base64
    import inspect
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import cv2
    import sweetviz as sv
    import xgboost as xgb
    from xgboost import XGBRegressor
    from xgboost import DMatrix
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from streamlit.components.v1 import html
    import sys
    import json
    import sweetviz as sv
    import seaborn as sns
    from pyecharts.charts import Bar
    from pyecharts.charts import Scatter  # 导入散点图模块
    from pyecharts.charts import Pie  # 导入饼图模块
    from pyecharts.globals import ChartType, SymbolType
    from pyecharts import options as opts  # 导入配置项
    from pyecharts.globals import ThemeType  # 主题配置项
    from pyecharts.globals import JsCode  # 可以用于执行Js代码
    import os
    from streamlit.components.v1 import html
    from streamlit_option_menu import option_menu
    from pyecharts.charts import Line
    from pyecharts.charts import Tab
    from pyecharts.charts import Bar
    from pyecharts.globals import ThemeType
    from pyecharts import options as opts
    from pyecharts.globals import CurrentConfig
    import matplotx
    from geopy.distance import geodesic
    from folium import plugins
    from folium.plugins import AntPath
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import time
    import torch
    from torch.utils.data import TensorDataset,DataLoader,random_split
    from torchvision import transforms
    import torch.nn.functional as F
    #import adaxgb
    import glob
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from PIL import ImageFile
    from PIL import Image
    from rich import print
    # 忽略损坏的图片
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    with st.sidebar:
        selected = option_menu(
            menu_title='垃圾分类',
            options=['Home', '数据展示','模型搭建', '图像识别', '意见反馈'],  # 每个选项名称
            icons=['house', 'graph-up', 'graph-up-arrow'],  # 每个选项图标
            menu_icon='cast',  # 标题旁边的图标
            default_index=0,  # 默认选项
            orientation='vertical',  # horizontal水平
            styles={
                "container": {"padding": "5!important",
                              "background-color": "#d3e3fd"},  # 调整菜单的Div容器
                "icon": {"color": "orange",
                         "font-size": "22px"},  # 调整图标样式
                "nav-link": {"font-size": "16px",
                             "font-weight": "bold",
                             "background-color": "#f7f8fd",  # 选项的背景颜色
                             "color": 'black'  # 选项的字体颜色
                             },  # 每个选项文本的样式
                "nav-link-selected": {"background-color": "#89e9f3",  # 选中选项的背景颜色
                                      "color": "red"  # 选中选项的字体颜色
                                      }  # 选中选项的样式
            },
        )
    with open('./data/infos.txt', 'r', encoding='utf-8') as f:
        infos = f.read()

    if selected == 'Home':
        # 使用 st.markdown 渲染 HTML 格式的内容
        st.markdown('# <div style="text-align: center; color: #0276fe;">基于深度学习的垃圾分类图像识别系统</div>', unsafe_allow_html=True)
        # 渲染文件中的内容，确保支持 HTML 标签
        st.markdown(f'## <div style="text-align: center; color: #3ee076;">{infos}</div>', unsafe_allow_html=True)

    if selected == 'Home':
        with st.sidebar:
            selected_sub = option_menu(
                key='cat',
                menu_title=None,
                options=['班级：人工智能1班', '小组：7组',
                         f'组长：郭金平'],
                icons=['house', 'clipboard2', 'airplane'],
                menu_icon='browser-firefox',
                default_index=0,
                orientation='vertical',
                styles={
                    "container": {"padding": "5!important",
                                  "background-color": "#ffffff"},
                    "icon": {"color": "orange", "font-size": "16px"},
                    "nav-link": {"font-size": "16px", "font-weight": "bold",
                                 "background-color": "#f7f8fd",
                                 "color": 'black'
                                 },
                    "nav-link-selected": {"background-color": "#89e9f3",
                                          "color": "orange"
                                          }
                }
            )
    if selected == '数据展示':
        st.title('部分数据集预览')
        tab1,tab2,tab3,tab4 = st.tabs(['Harmful', 'Kitchen', 'Recyclable', 'Other'])
        with tab1:
            image_paths = [
                './picture/harmful/h1.jpg',
                './picture/harmful/h2.jpg',
                './picture/harmful/h3.jpg',
                './picture/harmful/h4.jpg',
                './picture/harmful/h5.jpg',
                './picture/harmful/h6.jpg',
                './picture/harmful/h7.jpg',
                './picture/harmful/h8.jpg',
                './picture/harmful/h9.jpg',
                './picture/harmful/h10.jpg',
                './picture/harmful/h11.jpg',
                './picture/harmful/h12.jpg',
            ]
            columns_per_row=4
            for i in range(0, len(image_paths), columns_per_row):
                cols = st.columns(columns_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_paths):
                        col.image(image_paths[i + j], caption=f"Image {i + j + 1}", )
        with tab2:
            image_paths = [
                './picture/kitchen/k1.jpg',
                './picture/kitchen/k2.jpg',
                './picture/kitchen/k3.jpg',
                './picture/kitchen/k4.jpg',
                './picture/kitchen/k5.jpg',
                './picture/kitchen/k6.jpg',
                './picture/kitchen/k7.jpg',
                './picture/kitchen/k8.jpg',
                './picture/kitchen/k9.jpg',
                './picture/kitchen/k10.jpg',
                './picture/kitchen/k11.jpg',
                './picture/kitchen/k12.jpg',
            ]
            columns_per_row = 4
            for i in range(0, len(image_paths), columns_per_row):
                cols = st.columns(columns_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_paths):
                        col.image(image_paths[i + j], caption=f"Image {i + j + 1}", )
        with tab3:
            image_paths = [
                './picture/recyclable/r1.jpg',
                './picture/recyclable/r2.jpg',
                './picture/recyclable/r3.jpg',
                './picture/recyclable/r4.jpg',
                './picture/recyclable/r5.jpg',
                './picture/recyclable/r6.jpg',
                './picture/recyclable/r7.jpg',
                './picture/recyclable/r8.jpg',
                './picture/recyclable/r9.jpg',
                './picture/recyclable/r10.jpg',
                './picture/recyclable/r11.jpg',
                './picture/recyclable/r12.jpg',
            ]
            columns_per_row = 4
            for i in range(0, len(image_paths), columns_per_row):
                cols = st.columns(columns_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_paths):
                        col.image(image_paths[i + j], caption=f"Image {i + j + 1}", )
        with tab4:
            image_paths = [
                './picture/other/o1.jpg',
                './picture/other/o2.jpg',
                './picture/other/o3.jpg',
                './picture/other/o4.jpg',
                './picture/other/o5.jpg',
                './picture/other/o6.jpg',
                './picture/other/o7.jpg',
                './picture/other/o8.jpg',
                './picture/other/o9.jpg',
                './picture/other/o10.jpg',
                './picture/other/o11.jpg',
                './picture/other/o12.jpg',
            ]
            columns_per_row = 4
            for i in range(0, len(image_paths), columns_per_row):
                cols = st.columns(columns_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(image_paths):
                        col.image(image_paths[i + j], caption=f"Image {i + j + 1}")

    if selected == '模型搭建':
        with st.sidebar:
            selected_sub1 = option_menu(
                key='cat',
                menu_title=None,
                options=['VGG模型', 'AlexNet模型', 'ResNet模型', 'LeNet5模型'],
                icons=['house', 'clipboard2', 'airplane'],
                menu_icon='browser-firefox',
                default_index=0,
                orientation='vertical',
                styles={
                    "container": {"padding": "5!important",
                                  "background-color": "#ffffff"},
                    "icon": {"color": "orange", "font-size": "16px"},
                    "nav-link": {"font-size": "16px",
                                 "font-weight": "bold",
                                 "background-color": "#f7f8fd",
                                 "color": 'black'
                                 },
                    "nav-link-selected": {"background-color": "#89e9f3",
                                          "color": "orange"
                                          }
                }
            )
        if selected_sub1 == 'VGG模型':
            with st.sidebar:
                selected_sub = option_menu(
                    menu_title='作者信息',
                    options=['Author：李雅芳', '学号：2228724106', '点击打赏该作者'],
                    icons=['house', 'clipboard2', 'airplane'],
                    menu_icon='browser-firefox',
                    default_index=0,
                    orientation='vertical',
                    styles={
                        "container": {"padding": "5!important",
                                      "background-color": "#ffffff"},
                        "icon": {"color": "orange", "font-size": "16px"},
                        "nav-link": {"font-size": "16px", "font-weight": "bold",
                                     "background-color": "#f7f8fd",
                                     "color": 'black'
                                     },
                        "nav-link-selected": {"background-color": "#89e9f3",
                                              "color": "orange"
                                              }
                    }
                )
            if (selected_sub == '学号：2228724106') | (selected_sub == 'Author：李雅芳'):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(['VGG11', 'VGG13', 'VGG16', 'VGG19', '模型预测图像'])

                with tab1:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/VGG11折线图.png')
                    st.write('混淆矩阵')
                    st.image('./image/VGG11混淆矩阵.png')
                    st.write('分类报告')
                    st.image('./image/VGG11分类报告.png')

                with tab2:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/VGG13折线图.png')
                    st.write('混淆矩阵')
                    st.image('./image/VGG13混淆矩阵.png')
                    st.write('分类报告')
                    st.image('./image/VGG13分类报告.png')

                with tab3:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/VGG16折线图.png')
                    st.write('混淆矩阵')
                    st.image('./image/VGG16混淆矩阵.png')
                    st.write('分类报告')
                    st.image('./image/VGG16分类报告.png')

                with tab4:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/VGG19折线图.png')
                    st.write('混淆矩阵')
                    st.image('./image/VGG19混淆矩阵.png')
                    st.write('分类报告')
                    st.image('./image/VGG19分类报告.png')

                with tab5:
                    transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Resize((224, 224)),  # 调整图像大小
                        transforms.ToTensor(),  # 转为张量
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
                    ])

                    cfg = {
                        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                                  512,
                                  512, 'M'],
                    }


                    class VGG(nn.Module):
                        def __init__(self, vgg_name, num_classes):
                            super(VGG, self).__init__()
                            self.features = self._make_layers(cfg[vgg_name])
                            self.classifier = nn.Sequential(
                                nn.Linear(512 * 7 * 7, 512),  # 假设输入图像为 224x224，经过5次最大池化后特征图尺寸为 7x7
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),  # Dropout层,可以0.2概率丢弃一些神经元
                                nn.Linear(256, num_classes)  # 多分类任务
                            )

                        def forward(self, x):
                            out = self.features(x)  # 通过特征层提取
                            out = out.view(out.size(0), -1)
                            out = self.classifier(out)
                            out = F.log_softmax(out, dim=1)  # 添加softmax激活函数
                            return out

                        # 根据配置创建网络层
                        def _make_layers(self, cfg):
                            layers = []
                            in_channels = 3  # 输入的通道数
                            for x in cfg:
                                if x == 'M':  # 说明遇到了最大池化层
                                    layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
                                else:
                                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm2d(x),
                                               nn.ReLU()
                                               ]
                                    in_channels = x
                            return nn.Sequential(*layers)


                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    VGG11 = VGG(vgg_name='VGG11', num_classes=4).to(device)
                    VGG13 = VGG(vgg_name='VGG13', num_classes=4).to(device)
                    VGG16 = VGG(vgg_name='VGG16', num_classes=4).to(device)
                    VGG19 = VGG(vgg_name='VGG19', num_classes=4).to(device)


                    def predict_image(image_path, model, classes, device):
                        # 加载和预处理图像
                        image = Image.open(image_path).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
                        img = transform(image)
                        img = img.unsqueeze(0).to(device)

                        # 进行预测
                        model.eval()
                        with torch.no_grad():
                            output = model(img)
                        prob = F.softmax(output, dim=1)
                        value, predicted = torch.max(output.data, 1)

                        # 返回预测结果
                        return prob, predicted.item(), classes[predicted.item()]


                    # 设置设备
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # 创建模型实例并加载权重
                    model11 = VGG11
                    model13 = VGG13
                    model16 = VGG16
                    model19 = VGG19
                    # 加载模型的state_dict
                    state_dict11 = torch.load('./model/VGG11.pth', map_location=device)
                    model11.load_state_dict(state_dict11)
                    state_dict13 = torch.load('./model/VGG13.pth', map_location=device)
                    model13.load_state_dict(state_dict13)
                    state_dict16 = torch.load('./model/VGG16.pth', map_location=device)
                    model16.load_state_dict(state_dict16)
                    state_dict19 = torch.load('./model/VGG19.pth', map_location=device)
                    model19.load_state_dict(state_dict19)

                    # 设置设备
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model11 = model11.to(device)
                    model11.eval()  # 调整为评估模式
                    model13 = model13.to(device)
                    model13.eval()  # 调整为评估模式
                    model16 = model16.to(device)
                    model16.eval()  # 调整为评估模式
                    model19 = model19.to(device)
                    model19.eval()  # 调整为评估模式

                    # 定义类别
                    classes = ('Harmful', 'Kitchen', 'Other', 'Recyclable')
                    # 文件上传器
                    file_uploader = st.file_uploader("Choose an image...", type=["jpg", "png"])
                    if file_uploader is not None:
                        # 预测图像
                        results = []
                        for model_name, model in [('VGG11', model11), ('VGG13', model13), ('VGG16', model16),
                                                  ('VGG19', model19)]:
                            prob, predicted_index, pred_class = predict_image(file_uploader, model, classes, device)
                            results.append((model_name, prob, predicted_index, pred_class))

                        # 展示预测结果
                        st.image(file_uploader, caption='Uploaded Image', use_container_width=True)  # 如果需要展示上传的图像，可以取消注释这行代码
                        for model_name, prob, predicted_index, pred_class in results:
                            st.write(f"模型{model_name}的预测:")
                            for i, p in enumerate(prob[0], start=0):  # 假设 prob[0] 是一个包含所有概率的张量
                                st.write(f'预测为{classes[i]}垃圾概率为: {p:.4f}')
                            st.write(f"最终预测类别为: {pred_class}")
                    # 如果没有上传文件，则展示说明
                    else:
                        st.write("Please upload an image to see the prediction.")
            if selected_sub == '点击打赏该作者':
                st.image("./picture/感谢.gif", width=500)
        if selected_sub1 == 'AlexNet模型':
            with st.sidebar:
                selected_sub = option_menu(
                    menu_title='作者信息',
                    options=['Author：马文静', '学号：2228324096', '点击打赏该作者'],
                    icons=['house', 'clipboard2', 'airplane'],
                    menu_icon='browser-firefox',
                    default_index=0,
                    orientation='vertical',
                    styles={
                        "container": {"padding": "5!important",
                                      "background-color": "#ffffff"},
                        "icon": {"color": "orange", "font-size": "16px"},
                        "nav-link": {"font-size": "16px", "font-weight": "bold",
                                     "background-color": "#f7f8fd",
                                     "color": 'black'
                                     },
                        "nav-link-selected": {"background-color": "#89e9f3",
                                              "color": "orange"
                                              }
                    }
                )
            if (selected_sub == '学号：2228324096') | (selected_sub == 'Author：马文静'):
                tab1, tab2, tab3,tab4 = st.tabs(['Tab 1', 'Tab 2', 'Tab 3','Tab 4'])
                with tab1:
                    class AlexNet(nn.Module):
                        def __init__(self):
                            super(AlexNet, self).__init__()
                            self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2)
                            )
                            self.conv2 = nn.Sequential(
                                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2)
                            )
                            self.conv3 = nn.Sequential(
                                nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True)
                            )
                            self.flatten = nn.Flatten()
                            self.fc = nn.Sequential(
                                nn.Linear(9216, 4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 4)
                            )

                        def forward(self, x):
                            x = self.conv1(x)
                            x = self.conv2(x)
                            x = self.conv3(x)
                            x = self.flatten(x)
                            x = self.fc(x)
                            return x


                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = AlexNet().to(device)
                    st.write('主要代码展示')
                    st.image('./image/Alexnet主要代码展示.png')
                with tab2:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/AlexNet.png')
                with tab3:
                    st.write('混淆矩阵')
                    st.image('./image/AlexNet混淆矩阵.png')
                with tab4:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # 设置类别
                    classes = {0: 'Harmful', 1: 'Kitchen', 2: 'Other', 3: 'Recyclable'}

                    # resnet18 = resnet_plus(BasicBlock, [2, 2, 2, 2]).to(device)
                    # resnet18.load_state_dict(torch.load('./model/resnet18.pth', map_location=device, weights_only=True))
                    # resnet18.eval()

                    # 创建模型实例并加载权重
                    model = AlexNet()
                    state_dict = torch.load('./model/AlexNet.pth', map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()  # 调整为评估模式


                    # 图像预处理函数
                    def preprocess_image(image):
                        image = cv2.resize(image, (128, 128))  # 假设训练时使用224x224大小
                        transform = transforms.Compose([
                            transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0, 1]
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
                        ])
                        img = transform(image).unsqueeze(0)  # 增加批次维度
                        return img.to(device)


                    # 预测函数
                    def predict_image(image):
                        # 将图像处理为模型输入
                        img = preprocess_image(image)
                        # 模型评估模式
                        model.eval()
                        # 不跟踪梯度，减少内存和计算量
                        with torch.no_grad():
                            output = model(img)
                        # 应用softmax获取概率分布
                        prob = F.softmax(output, dim=1)
                        # 获取预测类别
                        _, predicted = torch.max(output, 1)
                        pred_class = classes[predicted.item()]
                        return prob, predicted.item(), pred_class


                    st.write("AlexNet垃圾分类预测")
                    uploaded_file = st.file_uploader("上传图片 (png/jpg 格式)", type=["png", "jpg"])
                    if uploaded_file is not None:
                        # 显示上传的图片
                        image = Image.open(uploaded_file)
                        st.image(image, caption="上传的图片", use_container_width=True)
                        # 将PIL图片转换为OpenCV格式进行处理
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # 记录预测开始的时间
                        start_time = time.time()
                        prob, class_idx, class_name = predict_image(image)
                        # 记录预测结束的时间并计算预测时间
                        end_time = time.time()
                        prediction_time = end_time - start_time
                        # 显示预测结果
                        st.write(f"预测类别: {class_name}")
                        st.write(f"预测时间: {prediction_time:.4f} 秒")  # 显示预测时间
            if selected_sub == '点击打赏该作者':
                st.image("./picture/感谢.gif", width=500)
        if selected_sub1 == 'ResNet模型':
            with st.sidebar:
                selected_sub = option_menu(
                    menu_title='作者信息',
                    options=['Author：韩梦阁', '学号：2228724025', '点击打赏该作者'],
                    icons=['house', 'clipboard2', 'airplane'],
                    menu_icon='browser-firefox',
                    default_index=0,
                    orientation='vertical',
                    styles={
                        "container": {"padding": "5!important",
                                      "background-color": "#ffffff"},
                        "icon": {"color": "orange", "font-size": "16px"},
                        "nav-link": {"font-size": "16px", "font-weight": "bold",
                                     "background-color": "#f7f8fd",
                                     "color": 'black'
                                     },
                        "nav-link-selected": {"background-color": "#89e9f3",
                                              "color": "orange"
                                              }
                    }
                )
            if (selected_sub == '学号：2228724025') | (selected_sub == 'Author：韩梦阁'):
                tab1, tab2, tab3 = st.tabs(['Tab 1', 'Tab 2', 'Tab 3'])
                with tab1:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/resnet18.png')
                with tab2:
                    st.write('混淆矩阵')
                    st.image('./image/resnet18混淆矩阵.png')
                with tab3:
                    class BasicBlock(nn.Module):
                        expansion = 1  #

                        def __init__(self, input_channels, output_channels, strides=1):
                            super(BasicBlock, self).__init__()  # 调用父类初始化方法
                            # 第一个卷基层,输入为input_channels,输出为num_channels
                            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                                                   padding=1, stride=strides, bias=False)
                            self.bn1 = nn.BatchNorm2d(output_channels)
                            self.relu = nn.ReLU(inplace=True)  # 允许原地修改,可以减少内存的使用

                            # 第一个卷基层,输入为num_channels,输出为num_channels
                            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1,
                                                   stride=1, bias=False
                                                   )
                            # 定义两个批量归一化层,
                            self.bn2 = nn.BatchNorm2d(output_channels)
                            # 创建一个快捷连接,如果输入输出不一致,使用1x1卷积改变输出
                            self.shortcut = nn.Sequential()
                            if strides != 1 or input_channels != self.expansion * output_channels:
                                self.shortcut = nn.Sequential(
                                    nn.Conv2d(input_channels, self.expansion * output_channels,
                                              kernel_size=1, stride=strides, bias=False),
                                    nn.BatchNorm2d(self.expansion * output_channels)
                                )

                        def forward(self, x):
                            output = self.relu(self.bn1(self.conv1(x)))  # 网络的第一层
                            output = self.bn2(self.conv2(output))  # 网络l第二层
                            output += self.shortcut(x)  # 将快捷连接的输出,加到第二个卷积的输出上
                            output = self.relu(output)
                            return output

                    # 基本网格搭建
                    class ResNet_plus(nn.Module):
                        def __init__(self, block, num_block, num_class=10, init_channels=3):
                            super(ResNet_plus, self).__init__()
                            self.input_channels = 64
                            self.features = nn.Sequential(
                                nn.Conv2d(init_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True)
                            )
                            # 构建残差层，每个层有多个残差块构建
                            self.layer1 = self._make_layer(block, 64, num_block[0], stride=2)
                            self.layer2 = self._make_layer(block, 128, num_block[1], stride=3)
                            self.layer3 = self._make_layer(block, 256, num_block[2], stride=4)
                            self.layer4 = self._make_layer(block, 512, num_block[3], stride=5)
                            self.avgpool = nn.AvgPool2d(kernel_size=2)
                            # 全连接层，输出10个概率分布
                            self.fc = nn.Linear(512 * block.expansion, num_class)

                        def _make_layer(self, block, output_channels, num_block, stride):
                            # 构建残差层,包含多个残差块
                            strides_ = [stride] + [1] * (num_block - 1)
                            layers = []
                            for stride in strides_:
                                layers.append(block(self.input_channels, output_channels, stride))
                                self.input_channels = output_channels * block.expansion
                            return nn.Sequential(*layers)

                        def forward(self, x):
                            out = self.features(x)
                            out = self.layer1(out)
                            out = self.layer2(out)
                            out = self.layer3(out)
                            out = self.layer4(out)
                            out = self.avgpool(out)
                            out = out.view(out.size(0), -1)  # 展开到一维
                            out = self.fc(out)
                            return out

                    def resnet_plus(block=BasicBlock, layers=[2, 2, 2, 2]):
                        return ResNet_plus(block, num_block=layers, num_class=4, init_channels=3)

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # 设置类别
                    classes = {0: 'Harmful', 1: 'Kitchen', 2: 'Other', 3: 'Recyclable'}

                    # resnet18 = resnet_plus(BasicBlock, [2, 2, 2, 2]).to(device)
                    # resnet18.load_state_dict(torch.load('./model/resnet18.pth', map_location=device, weights_only=True))
                    # resnet18.eval()

                    # 创建模型实例并加载权重
                    model = resnet_plus()
                    state_dict = torch.load('./model/res2.pth', map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()  # 调整为评估模式

                    # 图像预处理函数
                    def preprocess_image(image):
                        image = cv2.resize(image, (128, 128))  # 假设训练时使用224x224大小
                        transform = transforms.Compose([
                            transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0, 1]
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
                        ])
                        img = transform(image).unsqueeze(0)  # 增加批次维度
                        return img.to(device)

                    # 预测函数
                    def predict_image(image):
                        # 将图像处理为模型输入
                        img = preprocess_image(image)
                        # 模型评估模式
                        model.eval()
                        # 不跟踪梯度，减少内存和计算量
                        with torch.no_grad():
                            output = model(img)
                        # 应用softmax获取概率分布
                        prob = F.softmax(output, dim=1)
                        # 获取预测类别
                        _, predicted = torch.max(output, 1)
                        pred_class = classes[predicted.item()]
                        return prob, predicted.item(), pred_class

                    st.write("ResNet18垃圾分类预测")
                    uploaded_file = st.file_uploader("上传图片 (png/jpg 格式)", type=["png", "jpg"])
                    if uploaded_file is not None:
                        # 显示上传的图片
                        image = Image.open(uploaded_file)
                        st.image(image, caption="上传的图片", use_column_width=True)
                        # 将PIL图片转换为OpenCV格式进行处理
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # 记录预测开始的时间
                        start_time = time.time()
                        prob, class_idx, class_name = predict_image(image)
                        # 记录预测结束的时间并计算预测时间
                        end_time = time.time()
                        prediction_time = end_time - start_time
                        # 显示预测结果
                        st.write(f"预测类别: {class_name}")
                        st.write(f"预测时间: {prediction_time:.4f} 秒")  # 显示预测时间

            if selected_sub == '点击打赏该作者':
                st.image("./picture/感谢.gif", width=500)
        if selected_sub1 == 'LeNet5模型':
            with st.sidebar:
                selected_sub = option_menu(
                    menu_title='作者信息',
                    options=['Author：郭金平', '学号：2228724238', '点击打赏该作者'],
                    icons=['house', 'clipboard2', 'airplane'],
                    menu_icon='browser-firefox',
                    default_index=0,
                    orientation='vertical',
                    styles={
                        "container": {"padding": "5!important",
                                      "background-color": "#ffffff"},
                        "icon": {"color": "orange", "font-size": "16px"},
                        "nav-link": {"font-size": "16px", "font-weight": "bold",
                                     "background-color": "#f7f8fd",
                                     "color": 'black'
                                     },
                        "nav-link-selected": {"background-color": "#89e9f3",
                                              "color": "orange"
                                              }
                    }
                )
            if (selected_sub == '学号：2228724238') | (selected_sub == 'Author：郭金平'):
                tab1, tab2, tab3,tab4 = st.tabs(['Tab 1', 'Tab 2', 'Tab 3','Tab 4'])
                with tab1:
                    st.write('主要代码展示')
                    st.image('./image/lenet5网络搭建.png')
                    class LeNet5(nn.Module):
                        def __init__(self, num_class=4):
                            super(LeNet5, self).__init__()
                            self.features = nn.Sequential(
                                nn.Conv2d(3, 6, kernel_size=5),
                                nn.ReLU(),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(6, 16, kernel_size=5),
                                nn.ReLU(),
                                nn.AvgPool2d(kernel_size=2, stride=2)
                            )
                            self.flatten = nn.Flatten()
                            self.classifier = nn.Sequential(
                                nn.Linear(13456, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Linear(84, num_class),
                            )

                        def forward(self, x):
                            x = self.features(x)
                            x = self.flatten(x)
                            x = self.classifier(x)
                            return x

                with tab2:
                    st.write('训练过程中精确度和损失的变化情况')
                    st.image('./image/lenet5.png')
                with tab3:
                    st.write('混淆矩阵')
                    st.image('./image/lenet5混淆矩阵.png')

                with tab4:
                    st.write('模型预测图像')
                    # 封装成函数
                    def predict_image(image_path, model, classes, device):
                        # 加载和预处理图像
                        image = Image.open(image_path).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
                        img = transform(image)
                        img = img.unsqueeze(0).to(device)

                        # 进行预测
                        model.eval()
                        with torch.no_grad():
                            output = model(img)
                        prob = F.softmax(output, dim=1)
                        value, predicted = torch.max(output.data, 1)

                        # 返回预测结果
                        return prob, predicted.item(), classes[predicted.item()]
                    # 设置设备
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # 创建模型实例并加载权重
                    model = LeNet5()
                    state_dict = torch.load('./model/LeNet5-1.pth', map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()  # 调整为评估模式

                    # 定义类别
                    classes = ('Harmful', 'Kitchen', 'Other', 'Recyclable')

                    # 文件上传器
                    file_uploader = st.file_uploader("Choose an image...", type=["jpg", "png"])
                    if file_uploader is not None:
                        # 预测图像
                        prob, predicted_index, pred_class = predict_image(file_uploader, model, classes, device)

                        # 展示预测结果
                        st.image(file_uploader, caption='Uploaded Image', use_container_width=True)
                        for i, p in enumerate(prob[0], start=0):  # 假设 prob[0] 是一个包含所有概率的张量
                            st.write(f'预测为{classes[i]}垃圾概率为: {p:.4f}')
                        st.write(f'最终预测的类别索引为: {predicted_index}')
                        st.write(f'最终预测的类别为: {pred_class}')

                    # 如果没有上传文件，则展示说明
                    else:
                        st.write("Please upload an image to see the prediction.")

            if selected_sub == '点击打赏该作者':
                st.image("./picture/感谢.gif", width=500)
    if selected == '图像识别':
        with st.sidebar:
            selected_sub1 = option_menu(
                key='cat',
                menu_title=None,
                options=['垃圾分类', ],
                icons=['house', 'clipboard2', 'airplane'],
                menu_icon='browser-firefox',
                default_index=0,
                orientation='vertical',
                styles={
                    "container": {"padding": "5!important",
                                  "background-color": "#ffffff"},
                    "icon": {"color": "orange", "font-size": "16px"},
                    "nav-link": {"font-size": "16px",
                                 "font-weight": "bold",
                                 "background-color": "#f7f8fd",
                                 "color": 'black'
                                 },
                    "nav-link-selected": {"background-color": "#89e9f3",
                                          "color": "orange"
                                          }
                }
            )
        if selected_sub1 == '垃圾分类':
            with st.sidebar:
                selected_sub = option_menu(
                    menu_title='作者信息',
                    options=['Author：李泓苏', '学号：2228724177', '点击打赏该作者'],
                    icons=['house', 'clipboard2', 'airplane'],
                    menu_icon='browser-firefox',
                    default_index=0,
                    orientation='vertical',
                    styles={
                        "container": {"padding": "5!important",
                                      "background-color": "#ffffff"},
                        "icon": {"color": "orange", "font-size": "16px"},
                        "nav-link": {"font-size": "16px", "font-weight": "bold",
                                     "background-color": "#f7f8fd",
                                     "color": 'black'
                                     },
                        "nav-link-selected": {"background-color": "#89e9f3",
                                              "color": "orange"
                                              }
                    }
                )
            if selected_sub == '点击打赏该作者':
                st.image("./picture/感谢.gif", width=500)
            if (selected_sub == 'Author：李泓苏') | (selected_sub == '学号：2228724177'):
                tab1, tab2 = st.tabs(['🌏AlexNet', '📊VGG'])
                with tab1:
                    class AlexNet(nn.Module):
                        def __init__(self):
                            super(AlexNet, self).__init__()
                            self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2)
                            )
                            self.conv2 = nn.Sequential(
                                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2)
                            )
                            self.conv3 = nn.Sequential(
                                nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True)
                            )
                            self.flatten = nn.Flatten()
                            self.fc = nn.Sequential(
                                nn.Linear(9216, 4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 4)
                            )

                        def forward(self, x):
                            x = self.conv1(x)
                            x = self.conv2(x)
                            x = self.conv3(x)
                            x = self.flatten(x)
                            x = self.fc(x)
                            return x


                    # 设置类别
                    classes = {0: 'Harmful', 1: 'Kitchen', 2: 'Other', 3: 'Recyclable'}
                    # 创建模型实例并加载权重
                    model = AlexNet()
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    state_dict = torch.load('./model/AlexNet.pth', map_location=device)
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()  # 调整为评估模式


                    # 图像预处理函数
                    def preprocess_image(image):
                        image = cv2.resize(image, (128, 128))  # 假设训练时使用224x224大小
                        transform = transforms.Compose([
                            transforms.ToTensor(),  # 转换为Tensor并自动归一化到[0, 1]
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
                        ])
                        img = transform(image).unsqueeze(0)  # 增加批次维度
                        return img.to(device)


                    # 预测函数
                    def predict_image(image):
                        # 将图像处理为模型输入
                        img = preprocess_image(image)
                        # 模型评估模式
                        model.eval()
                        # 不跟踪梯度，减少内存和计算量
                        with torch.no_grad():
                            output = model(img)
                        # 应用softmax获取概率分布
                        prob = F.softmax(output, dim=1)
                        # 获取预测类别
                        _, predicted = torch.max(output, 1)
                        pred_class = classes[predicted.item()]
                        return prob, predicted.item(), pred_class



                    uploaded_file = st.file_uploader("上传图片 (png/jpg 格式)", type=["png", "jpg"])
                    if uploaded_file is not None:
                        # 显示上传的图片
                        image = Image.open(uploaded_file)
                        st.image(image, caption="上传的图片", use_container_width=True)
                        # 将PIL图片转换为OpenCV格式进行处理
                        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # 记录预测开始的时间
                        start_time = time.time()
                        prob, class_idx, class_name = predict_image(image)
                        # 记录预测结束的时间并计算预测时间
                        end_time = time.time()
                        prediction_time = end_time - start_time
                        # 显示预测结果
                        st.write(f"预测类别: {class_name}")
                        st.write(f"预测时间: {prediction_time:.4f} 秒")  # 显示预测时间
                with tab2:
                    import streamlit as st
                    import torch
                    from torch import nn
                    import torch.nn.functional as F
                    from torchvision import transforms
                    from PIL import Image


                    # 定义VGG模型
                    class VGG(nn.Module):
                        def __init__(self, vgg_name, num_classes):
                            super(VGG, self).__init__()
                            self.features = self._make_layers(cfg[vgg_name])
                            self.classifier = nn.Sequential(
                                nn.Linear(512 * 7 * 7, 512),  # 假设输入图像为 224x224，经过5次最大池化后特征图尺寸为 7x7
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),  # Dropout层,可以0.2概率丢弃一些神经元
                                nn.Linear(256, num_classes)  # 多分类任务
                            )

                        def forward(self, x):
                            out = self.features(x)  # 通过特征层提取
                            out = out.view(out.size(0), -1)
                            out = self.classifier(out)
                            out = F.log_softmax(out, dim=1)  # 添加softmax激活函数
                            return out

                        # 根据配置创建网络层
                        def _make_layers(self, cfg):
                            layers = []
                            in_channels = 3  # 输入的通道数
                            for x in cfg:
                                if x == 'M':  # 说明遇到了最大池化层
                                    layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
                                else:
                                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm2d(x),
                                               nn.ReLU()
                                               ]
                                    in_channels = x
                            return nn.Sequential(*layers)


                    # 模型配置
                    cfg = {
                        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
                                  512, 512, 'M'],
                    }

                    # 加载模型
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # 定义类别
                    classes = ('Harmful', 'Kitchen', 'Other', 'Recyclable')


                    # 定义预测函数
                    def predict_image(image_path, model, classes, device):
                        # 加载和预处理图像
                        image = Image.open(image_path).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
                        img = transform(image)
                        img = img.unsqueeze(0).to(device)

                        # 进行预测
                        model.eval()
                        with torch.no_grad():
                            output = model(img)
                        prob = F.softmax(output, dim=1)
                        value, predicted = torch.max(output.data, 1)

                        # 返回预测结果
                        return prob, predicted.item(), classes[predicted.item()]


                    # 创建模型实例
                    model11 = VGG(vgg_name='VGG11', num_classes=4).to(device)
                    model13 = VGG(vgg_name='VGG13', num_classes=4).to(device)
                    model16 = VGG(vgg_name='VGG16', num_classes=4).to(device)
                    model19 = VGG(vgg_name='VGG19', num_classes=4).to(device)

                    # 加载模型的权重
                    state_dict11 = torch.load('./model/VGG11.pth', map_location=device)
                    model11.load_state_dict(state_dict11)
                    state_dict13 = torch.load('./model/VGG13.pth', map_location=device)
                    model13.load_state_dict(state_dict13)
                    state_dict16 = torch.load('./model/VGG16.pth', map_location=device)
                    model16.load_state_dict(state_dict16)
                    state_dict19 = torch.load('./model/VGG19.pth', map_location=device)
                    model19.load_state_dict(state_dict19)

                    # 设置模型为评估模式
                    model11.eval()
                    model13.eval()
                    model16.eval()
                    model19.eval()

                    # 文件上传器

                    file_uploader = st.file_uploader("上传图片 (png/jpg 格式)", type=["jpg", "png"])

                    if file_uploader is not None:
                        # 选择模型
                        model_choice = st.selectbox("选择模型", ['VGG11', 'VGG13', 'VGG16', 'VGG19'])

                        # 选择对应的模型
                        if model_choice == 'VGG11':
                            model = model11
                        elif model_choice == 'VGG13':
                            model = model13
                        elif model_choice == 'VGG16':
                            model = model16
                        else:
                            model = model19

                        # 预测图像
                        prob, predicted_index, pred_class = predict_image(file_uploader, model, classes, device)

                        # 展示预测结果
                        st.image(file_uploader, caption='显示图片', use_container_width=True)  # 展示上传的图像
                        st.write(f"模型{model_choice}的预测:")
                        for i, p in enumerate(prob[0], start=0):  # 假设 prob[0] 是一个包含所有概率的张量
                            st.write(f'预测为{classes[i]}垃圾概率为: {p:.4f}')
                        st.write(f"最终预测类别为: {pred_class}")
                    else:
                        st.write("Please upload an image to see the prediction.")

    if selected == '意见反馈':
        def get_image_base64(image_path):
            with open('./picture/反馈.gif', "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
            return encoded
        st.markdown("<h2 style='text-align: center; color: #4CAF50;'>意见反馈</h2>", unsafe_allow_html=True)
        # 使用表单
        with st.form('my_form'):
            name = st.text_input(label='请输入你的名字 👤')
            email = st.text_input(label='请输入你的邮箱 📧')
            content = st.text_area(label='请输入你的反馈意见 ✍️')

            submit = st.form_submit_button(
                '提交',
                type="primary"  # 设置按钮风格
            )
            if submit:
                # GIF 图片路径
                gif_path = r"C:\Users\李泓苏\Desktop\小组作业\picture\反馈.gif"

                try:
                    gif_base64 = get_image_base64(gif_path)
                    st.markdown(
                        f"""
                            <div style="text-align: center;">
                                <img src="data:image/gif;base64,{gif_base64}" alt="反馈成功" width="400" style="border-radius: 10px;">
                                <h3 style="color: #4CAF50;">感谢您的反馈! 🙏</h3>
                            </div>
                            """,
                        unsafe_allow_html=True
                    )
                except FileNotFoundError:
                    st.error("GIF 图片未找到，请检查路径！")