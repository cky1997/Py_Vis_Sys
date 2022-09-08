from functools import reduce
from tqdm import tqdm
import time
import matplotlib.patches as mpatches
from PyQt5.QtCore import pyqtSignal, QObject
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox, \
    QAbstractItemView, QVBoxLayout, QLabel
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIntValidator
import pandas as pd
import numpy as np
import sys
from threading import Thread

# import sysInfo
from mainwin import Ui_Form
import os
import random
import tools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlibwidget import MatplotlibWidget
import time


# class ThreadClass(QtCore.QThread):
#
#     progress = QtCore.pyqtSignal(object)
#
#     def __init__(self, parent=None):
#         super(ThreadClass, self).__init__(parent)
#
#
#     def run(self):
#         t = tqdm(range(100))
#         for e in t:
#             self.progress.emit(e)
#             time.sleep(0.1)
#
#     def stop(self):
#         # self.isRunning = False
#         self.terminate()
#
#         # class MainWindow(QWidget, Ui_Form):
class Mysignals(QObject):
    text_print = pyqtSignal(str)


class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # self.pushButton_6.clicked.connect(self.crawlClicked)
        self.ms = Mysignals()
        self.ms.text_print.connect(self.printToGui)

        self.init_widget()

        # self.wd_1.clicked.connect(self.dosomestuff)
        self.wd_1.mouseDoubleClickSignal.connect(self.lbl_double_click1)
        self.wd_2.mouseDoubleClickSignal.connect(self.lbl_double_click2)
        self.wd_3.mouseDoubleClickSignal.connect(self.lbl_double_click3)
        self.wd_4.mouseDoubleClickSignal.connect(self.lbl_double_click4)
        self.wd_5.mouseDoubleClickSignal.connect(self.lbl_double_click5)

        self.lineEdit.setPlaceholderText(" keywords")
        self.lineEdit_2.setPlaceholderText(" id: 17283~73469")
        # 整数校验器[1, 99] 精度小数点后两位
        intvalidator = QIntValidator()
        intvalidator.setRange(17283, 73469)
        self.lineEdit_2.setValidator(intvalidator)

        # 加载原始数据
        self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
        self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
        self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

        self.all_text_content = self.all_data['content']
        self.vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True, stop_words='english',
                                      max_df=0.3)
        self.vector_fit = self.vector.fit_transform(self.all_text_content)
        print(self.vector_fit)

        # 加载处理后的数据
        self.all_pro_data = pd.read_csv('/Users/chenkanyu/Desktop/code/processed_dataset.csv')
        self.all_pro_data = self.all_pro_data[pd.isna(self.all_pro_data['title']) == False]
        self.all_pro_data = self.all_pro_data[pd.isna(self.all_pro_data['content']) == False]

        self.all_pro_text_content = self.all_pro_data['content']
        #
        # self.vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", max_df=0.3, lowercase=True, stop_words='english')
        # self.vector_fit = self.vector.fit_transform(self.all_pro_text_content)

        self.pushButton.clicked.connect(self.dataHead)
        self.pushButton_2.clicked.connect(self.indi_wordcloud)
        self.pushButton_3.clicked.connect(self.search_articles)
        self.pushButton_4.clicked.connect(self.find_similar_articles)
        self.pushButton_5.clicked.connect(self.multi_wordcloud)
        # self.pushButton_6.clicked.connect(self.start_thread)
        # self.pushButton_7.clicked.connect(self.stop_worker)

        self.tableWidget.itemDoubleClicked.connect(self.detailed_data)

        # self.wd_1.mouseDoubleClickEvent()
        # self.wd_1 = MyLabel(self)
        # self.wd_1.mouseDoubleClickEvent(self,print("aaa"))

        # 一组词云的文件路径
        self.wc_file_path = []
        self.wc_file_id = []

        self.labels = [self.wd_1, self.wd_2, self.wd_3, self.wd_4, self.wd_5]

    def printToGui(self, text):
        self.textEdit.append(str(text))

    # def crawlClicked(self):
    #     def run():
    #         for i in range(1, 6):
    #             self.ms.text_print.emit(f"item-{i}")
    #             time.sleep(1)
    #     t = Thread(target=run)
    #     t.start()

    # def start_thread(self):
    #     self.thread = ThreadClass()
    #     self.thread.progress.connect(self.update_progress)
    #     self.thread.start()
    #
    # def update_progress(self, progress):
    #     self.progressBar.setValue(progress)
    def multi_wordcloud(self):

        # Clear textEdit
        self.textEdit.clear()

        # Clear plot area
        self.matplotlibwidget.ax.clear()
        self.matplotlibwidget.ax.axis("off")


        # Clear label
        for i in self.labels:
            i.clear()
        self.wc_file_path = []
        # self.wc_file_path = ["aaaa"]

        # selected_id_lst = [53775, 53892]
        selected_id_lst = []
        for idx in self.tableWidget.selectionModel().selectedRows():
            selected_id_lst.append(int(self.tableWidget.item(idx.row(), 0).text()))

        print(selected_id_lst)

        if len(selected_id_lst) < 2:
            QMessageBox.critical(self, "error", "Please select at least two articles.")
        else:
            # # Add legend according to the number of articles
            # self.plot_matwidget(len(selected_id_lst))

            # Get the id of selected articles
            self.wc_file_id = selected_id_lst
            print("self.wc_file_id" + str(self.wc_file_id))

            coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
            # text = []
            selected_count_df = []
            for i in range(len(selected_id_lst)):
                text = []
                # 某一篇的文章内容
                # text_content = self.all_pro_data[self.all_pro_data['id'].isin(selected_id_lst)].iloc[i]["content"]
                text_content = self.all_pro_data[self.all_pro_data['id'] == selected_id_lst[i]].iloc[0]["content"]
                text.append(text_content)
                count_matrix = coun_vect.fit_transform(text)
                count_array = count_matrix.toarray()
                # selected_count_array.append(count_array)
                df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
                text_dict = df.iloc[0].sort_values(ascending=False)[:80]
                # data = count_array, columns = coun_vect.get_feature_names_out()
                text_dict_df = pd.DataFrame({'words': text_dict.index, 'frequency': text_dict.values})
                selected_count_df.append(text_dict_df)
                # 这里可以考虑
                # selected_count_df.append(text_dict_df[text_dict_df['frequency']>3])

            print(selected_count_df)
            inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='outer', suffixes=(None, '_x')),
                                     selected_count_df)
            # print(inter_text_dict.iloc[:, 1:])

            # print(inter_text_dict)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            inter_text_dict["appear_count"] = (inter_text_dict.iloc[:, 1:].notnull()).astype(int).sum(axis=1)
            inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:-1].sum(axis=1)

            print("inter_text_dict" + str(inter_text_dict))
            print(type(inter_text_dict))
            # inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
            # print("inter_text_dict"+str(inter_text_dict))
            #
            # 获得共同词， 返回list
            inter_text_lst = np.array(inter_text_dict.iloc[:]["words"]).tolist()
            print("inter_text_lst" + str(inter_text_lst))

            def run():
                for j in range(len(selected_id_lst)):
                    # text_x_is_in_common = [1 if selected_count_df[j].iloc[i]["words"] in inter_text_lst else 0 for i in
                    #                       range(selected_count_df[j].shape[0])]
                    # selected_count_df[j]["is_in_common"] = pd.DataFrame(text_x_is_in_common, columns=['is_in_common'])
                    selected_count_df[j] = pd.merge(selected_count_df[j], inter_text_dict[["words", "appear_count"]],
                                                    on=["words"], how='left', suffixes=(None, '_x'))
                    # text1_fre_df.groupby("is_in_common")["is_in_common"]类型为Series
                    print(selected_count_df[j])

                    selected_count_df[j]["color"] = selected_count_df[j].groupby("appear_count")["appear_count"]. \
                        transform(lambda x: "#9400D3" if (x == 1).all() else ("#00688B" if (x == 2).all() else (
                        "#FFB90F" if (x == 3).all() else ("#CD9B9B" if (x == 4).all() else "#FF0000"))))

                    txt_x_color_to_words = selected_count_df[j].groupby("color")["words"].agg(list).to_dict()

                    print("-" * 20)
                    print(j)
                    print(txt_x_color_to_words)
                    print("-" * 20)

                    txt_x_fre_df_wd = pd.Series(selected_count_df[j]['frequency'].values,
                                                index=selected_count_df[j].iloc[:]["words"])
                    txt_x_fre_df_wd = txt_x_fre_df_wd.sort_values(ascending=False)[:80]

                    # ax = fig.add_subplot(1, len(selected_id_lst), i + 1)

                    wordcloud = WordCloud(width=500,
                                          height=500,
                                          max_words=80,
                                          min_word_length=3,
                                          prefer_horizontal=0.7,
                                          scale=30,
                                          background_color="rgba(255, 255, 255, 0)",
                                          mode="RGBA").generate_from_frequencies(txt_x_fre_df_wd)
                    # mapping color to words, type dict
                    # print(text1_fre_df.groupby("color")["words"].agg(list).to_dict())

                    default_color = 'grey'

                    # Create a color function with single tone
                    customed_color_func = tools.SimpleGroupedColorFunc(txt_x_color_to_words, default_color)

                    # Apply our color function
                    wordcloud.recolor(color_func=customed_color_func)

                    # Prompt
                    self.ms.text_print.emit(f"word cloud {j+1} generated...")

                    wordcloud.to_file(
                        os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[j]) + ".png"))
                    # Prompt
                    self.ms.text_print.emit(f"word cloud {j+1} saved...")
                    self.wc_file_path.append(
                        "/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[j]) + ".png")
                    print(selected_id_lst[j])
                    print(self.wc_file_path)
                    # time.sleep(1)

                for i in range(len(self.wc_file_path)):
                    # jpg = QPixmap(file_name[i]).scaled(self.labels[i].width(), self.labels[i].height())
                    jpg = QPixmap(self.wc_file_path[i])
                    self.labels[i].setPixmap(jpg)
                    self.labels[i].setScaledContents(True)
                    # Prompt
                    self.ms.text_print.emit(f"word cloud {i+1} displayed...")


                # Add legend according to the number of articles
                self.plot_matwidget(len(selected_id_lst))
                self.ms.text_print.emit(f"word clouds saved sucessfully.")

                

            t = Thread(target=run)
            t.start()
            # 设置新列，确认是否为共同词，是 改为1
            # for j in range(len(selected_id_lst)):
            #     # text_x_is_in_common = [1 if selected_count_df[j].iloc[i]["words"] in inter_text_lst else 0 for i in
            #     #                       range(selected_count_df[j].shape[0])]
            #     # selected_count_df[j]["is_in_common"] = pd.DataFrame(text_x_is_in_common, columns=['is_in_common'])
            #     selected_count_df[j] = pd.merge(selected_count_df[j], inter_text_dict[["words", "appear_count"]],
            #                                     on=["words"], how='left', suffixes=(None, '_x'))
            #     # text1_fre_df.groupby("is_in_common")["is_in_common"]类型为Series
            #     print(selected_count_df[j])
            #
            #     selected_count_df[j]["color"] = selected_count_df[j].groupby("appear_count")["appear_count"]. \
            #         transform(lambda x: "#9400D3" if (x == 1).all() else ("#00688B" if (x == 2).all() else (
            #         "#FFB90F" if (x == 3).all() else ("#CD9B9B" if (x == 4).all() else "#FF0000"))))
            #
            #     txt_x_color_to_words = selected_count_df[j].groupby("color")["words"].agg(list).to_dict()
            #
            #     print("-" * 20)
            #     print(j)
            #     print(txt_x_color_to_words)
            #     print("-" * 20)
            #
            #     txt_x_fre_df_wd = pd.Series(selected_count_df[j]['frequency'].values,
            #                                 index=selected_count_df[j].iloc[:]["words"])
            #     txt_x_fre_df_wd = txt_x_fre_df_wd.sort_values(ascending=False)[:80]
            #
            #     # ax = fig.add_subplot(1, len(selected_id_lst), i + 1)
            #
            #     wordcloud = WordCloud(width=500,
            #                           height=500,
            #                           max_words=80,
            #                           min_word_length=3,
            #                           prefer_horizontal=0.7,
            #                           scale=30,
            #                           background_color="rgba(255, 255, 255, 0)",
            #                           mode="RGBA").generate_from_frequencies(txt_x_fre_df_wd)
            #     # mapping color to words, type dict
            #     # print(text1_fre_df.groupby("color")["words"].agg(list).to_dict())
            #
            #     default_color = 'grey'
            #
            #     # Create a color function with single tone
            #     customed_color_func = tools.SimpleGroupedColorFunc(txt_x_color_to_words, default_color)
            #
            #     # Apply our color function
            #     wordcloud.recolor(color_func=customed_color_func)
            #
            #     wordcloud.to_file(
            #         os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[j]) + ".png"))
            #     self.wc_file_path.append("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[j]) + ".png")
            #     print(selected_id_lst[j])
            #     print(self.wc_file_path)

            # for i in range(len(self.wc_file_path)):
            #     # jpg = QPixmap(file_name[i]).scaled(self.labels[i].width(), self.labels[i].height())
            #     jpg = QPixmap(self.wc_file_path[i])
            #     self.labels[i].setPixmap(jpg)
            #     self.labels[i].setScaledContents(True)
            #
            # self.label.setText("word clouds saved sucessfully.")

    def init_widget(self):
        self.matplotlibwidget = MatplotlibWidget()
        self.layoutvertical = QVBoxLayout(self.widget)
        self.layoutvertical.addWidget(self.matplotlibwidget)

    # def testt(self):
    #     selected_id_lst = []
    #     for idx in self.tableWidget.selectionModel().selectedRows():
    #         selected_id_lst.append(int(self.tableWidget.item(idx.row(), 0).text()))
    #     self.plot_matwidget(len(selected_id_lst))

    def plot_matwidget(self, num_of_articles):

        self.matplotlibwidget.ax.clear()
        self.matplotlibwidget.ax.axis("off")

        darkviolet_patch = mpatches.Patch(color='#9400D3', label='Words in 1 articles')
        deepskyblue4_patch = mpatches.Patch(color='#00688B', label='Words in 2 articles')
        darkgoldenrod1_patch = mpatches.Patch(color='#FFB90F', label='Words in 3 articles')
        rosybrown3_patch = mpatches.Patch(color='#CD9B9B', label='Words in 4 articles')
        red_patch = mpatches.Patch(color='#FF0000', label='Words in 5 articles')
        patch_lst = [darkviolet_patch, deepskyblue4_patch, darkgoldenrod1_patch, rosybrown3_patch, red_patch]

        self.matplotlibwidget.ax.legend(
            handles=patch_lst[:num_of_articles],
            bbox_to_anchor=(1.15, 0.9), loc='upper right')

        self.matplotlibwidget.canvas.draw()

        # x = np.random.random(10)
        # y = np.random.random(10)
        # txts = ["P1", "P2", "P3", "P4", "P5",
        #        "P6", "P7", "P8", "P9", "P10",]
        #
        # self.matplotlibwidget.ax.scatter(x, y)
        # for index, txt in enumerate(txts):
        #
        #     self.matplotlibwidget.ax.annotate(txt, (x[index], y[index]))
        #
        #     seagreen1_patch = mpatches.Patch(color='#54FF9F', label='Words in 1 articles')
        #     turquoise1_patch = mpatches.Patch(color='#00F5FF', label='Words in 2 articles')
        #     darkgoldenrod1_patch = mpatches.Patch(color='#FFB90F', label='Words in 3 articles')
        #     rosybrown3_patch = mpatches.Patch(color='#CD9B9B', label='Words in 4 articles')
        #     red_patch = mpatches.Patch(color='#FF0000', label='Words in 5 articles')
        #     self.matplotlibwidget.ax.legend(handles=[seagreen1_patch, turquoise1_patch, darkgoldenrod1_patch, rosybrown3_patch, red_patch], bbox_to_anchor=(1.15, 1),  loc='upper right')
        #
        #     self.matplotlibwidget.canvas.draw()

    def lbl_double_click1(self):
        if len(self.wc_file_id) < 1:
            pass
        else:

            self.label.setPixmap(
                QPixmap(self.wc_file_path[0]))
            self.label.setScaledContents(True)

            self.selected_data = self.all_data[self.all_data['id'].isin([self.wc_file_id[0]])]
            # print(self.searched_data)

            self.textEdit.setPlainText(
                "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
                .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                        self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                        self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

    def lbl_double_click2(self):
        if len(self.wc_file_id) < 2:
            pass
        else:
            self.label.setPixmap(
                QPixmap(self.wc_file_path[1]))
            self.label.setScaledContents(True)

            self.selected_data = self.all_data[self.all_data['id'].isin([self.wc_file_id[1]])]
            # print(self.searched_data)

            self.textEdit.setPlainText(
                "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
                .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                        self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                        self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

    def lbl_double_click3(self):
        if len(self.wc_file_id) < 3:
            pass
        else:
            self.label.setPixmap(
                QPixmap(self.wc_file_path[2]))
            self.label.setScaledContents(True)

            self.selected_data = self.all_data[self.all_data['id'].isin([self.wc_file_id[2]])]
            # print(self.searched_data)

            self.textEdit.setPlainText(
                "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
                .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                        self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                        self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

    def lbl_double_click4(self):
        if len(self.wc_file_id) < 4:
            pass
        else:
            self.label.setPixmap(
                QPixmap(self.wc_file_path[3]))
            self.label.setScaledContents(True)

            self.selected_data = self.all_data[self.all_data['id'].isin([self.wc_file_id[3]])]
            # print(self.searched_data)

            self.textEdit.setPlainText(
                "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
                .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                        self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                        self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

    def lbl_double_click5(self):
        if len(self.wc_file_id) < 5:
            pass
        else:
            self.label.setPixmap(
                QPixmap(self.wc_file_path[4]))
            self.label.setScaledContents(True)

            self.selected_data = self.all_data[self.all_data['id'].isin([self.wc_file_id[4]])]
            # print(self.searched_data)

            self.textEdit.setPlainText(
                "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
                .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                        self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                        self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

    def find_similar_articles(self):
        self.tableWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        total_id = []
        for i in range(self.all_data.shape[0]):
            total_id.append(self.all_data.iloc[i]["id"])
        # total_id = self.all_data["id"]
        # print(total_id)
        requested_id = self.lineEdit_2.text()
        # requested_id_int = int(requested_id)
        if len(requested_id) == 0:
            QMessageBox.critical(self, "error", "ID cannot be null.")
        elif int(requested_id) in total_id:

            index = self.all_data[self.all_data["id"] == int(requested_id)].index.tolist()[0]
            print(index)

            indices = tools.SimilarityTools.find_similar(self.vector_fit, index)
            print(indices)

            self.searched_similar_data = self.all_data.iloc[indices, 1:3]

            NumRows = len(indices)

            self.tableWidget.setColumnCount(len(self.searched_similar_data.columns))
            self.tableWidget.setRowCount(NumRows)
            self.tableWidget.setHorizontalHeaderLabels(self.searched_similar_data.columns)
            self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView { font-size: 20pt; "
                                                              "background-color:rgb(240, 248, 255)}")
            self.tableWidget.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
            self.tableWidget.setStyleSheet("alternate-background-color: rgb(220, 220, 220);"
                                           "background-color: rgb(245, 245, 245);")

            for i in range(NumRows):
                for j in range(len(self.searched_similar_data.columns)):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.searched_similar_data.iat[i, j])))

            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()
        else:
            QMessageBox.critical(self, "error", "ID does not exist.")

        # self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
        # self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
        # self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

        # self.all_text_content = self.all_data['content']
        # vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True, stop_words='english',
        #                          max_df=0.3)
        # vector_fit = vector.fit_transform(self.all_text_content)

        # print(requested_id)
        # print(type(requested_id))

        # indices = Func_implement.find_similar(self.vector_fit, requested_id)
        # # print(indices)
        #
        # self.searched_similar_data = self.all_data.iloc[indices, 1:3]
        #
        # NumRows = len(indices)
        #
        # self.tableWidget.setColumnCount(len(self.searched_similar_data.columns))
        # self.tableWidget.setRowCount(NumRows)
        # self.tableWidget.setHorizontalHeaderLabels(self.searched_similar_data.columns)
        # self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView { font-size: 20pt; "
        #                                                   "background-color:rgb(240, 248, 255)}")
        # self.tableWidget.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        # self.tableWidget.setStyleSheet("alternate-background-color: rgb(220, 220, 220);"
        #                                "background-color: rgb(245, 245, 245);")
        #
        # for i in range(NumRows):
        #     for j in range(len(self.searched_similar_data.columns)):
        #         self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.searched_similar_data.iat[i, j])))
        #
        # self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()

    def search_articles(self):
        # self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
        # self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
        # self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

        # self.all_text_content = self.all_data['content']
        # vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True, stop_words='english',
        #                          max_df=0.3)
        # vector_fit = vector.fit_transform(self.all_text_content)
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)

        request = self.lineEdit.text()
        print(type(request))

        if len(request) == 0:
            QMessageBox.critical(self, "error", "Keywords cannot be null.")
        else:

            indices = tools.SimilarityTools.search(self.vector_fit, self.vector, request)

            self.searched_data = self.all_data.iloc[indices, 1:3]

            NumRows = len(indices)

            self.tableWidget.setColumnCount(len(self.searched_data.columns))
            self.tableWidget.setRowCount(NumRows)
            self.tableWidget.setHorizontalHeaderLabels(self.searched_data.columns)
            self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView { font-size: 20pt; "
                                                              "background-color:rgb(240, 248, 255)}")
            self.tableWidget.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
            self.tableWidget.setStyleSheet("alternate-background-color: rgb(220, 220, 220);"
                                           "background-color: rgb(245, 245, 245);")

            # text_content = self.all_data['content']

            # Func_implement.print_result(request, result, self.all_data)

            # self.textEdit.setPlainText(Func_implement.print_result(request, result, self.all_data))

            for i in range(NumRows):
                for j in range(len(self.searched_data.columns)):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.searched_data.iat[i, j])))

            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()

    def indi_wordcloud(self):

        selected_id_lst = []
        for idx in self.tableWidget.selectionModel().selectedRows():
            selected_id_lst.append(int(self.tableWidget.item(idx.row(), 0).text()))

        coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))

        # print(selected_id_lst)
        if len(selected_id_lst) == 0:
            QMessageBox.critical(self, "error", "Please select article(s).")
        elif len(selected_id_lst) == 1:

            text = []
            # 某一篇的文章内容
            text_content = self.all_pro_data[self.all_pro_data['id'].isin(selected_id_lst)].iloc[0]["content"]
            text.append(text_content)
            count_matrix = coun_vect.fit_transform(text)
            count_array = count_matrix.toarray()
            # selected_count_array.append(count_array)
            df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
            print(df)
            text_dict = df.iloc[0].sort_values(ascending=False)[:80]

            wordcloud = WordCloud(width=500,
                                  height=500,
                                  max_words=50,
                                  min_word_length=3,
                                  prefer_horizontal=0.7,
                                  scale=30,
                                  background_color="rgba(255, 255, 255, 0)",
                                  mode="RGBA").generate_from_frequencies(text_dict)
            # plt.imshow(wordcloud, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()

            wordcloud.to_file(
                os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png"))
            print("success")

            self.label.setPixmap(
                QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png")))
            self.label.setScaledContents(True)
        else:
            # text = []
            selected_count_df = []
            for i in range(len(selected_id_lst)):
                text = []
                # 某一篇的文章内容
                text_content = self.all_pro_data[self.all_pro_data['id'].isin(selected_id_lst)].iloc[i]["content"]
                text.append(text_content)
                count_matrix = coun_vect.fit_transform(text)
                count_array = count_matrix.toarray()
                # selected_count_array.append(count_array)
                df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
                print(df)
                text_dict = df.iloc[0].sort_values(ascending=False)[:80]
                print(type(text_dict))
                print(text_dict)
                # data = count_array, columns = coun_vect.get_feature_names_out()
                text_dict_df = pd.DataFrame({'words': text_dict.index, 'frequency': text_dict.values})
                selected_count_df.append(text_dict_df)

                # print(type(count_array))
                # print(text_dict_df)
                # print("-"*20)

            # print(selected_count_df)
            inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')),
                                     selected_count_df)
            # print(inter_text_dict.iloc[:, 1:])
            inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
            # inter_text_dict.loc['frequency'] = inter_text_dict.iloc[:, 1:].sum(axis=1)
            # inter_text_dict = inter_text_dict.iloc[:1, :]
            common_freq = inter_text_dict[["words", "frequency_total"]]
            # print(inter_text_dict.iloc[:, :2])
            print(common_freq)
            # print("-"*10)
            # print(common_freq.iloc[:]["words"])
            # print(type(common_freq.iloc[:]["words"]))
            # print("-"*10)
            common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])

            common_text_dict = common_freq_series.sort_values(ascending=False)[:80]
            print(common_text_dict)
            common_text_dict = common_text_dict[common_text_dict > 3]

            wordcloud = WordCloud(width=500,
                                  height=500,
                                  max_words=50,
                                  min_word_length=3,
                                  prefer_horizontal=0.7,
                                  scale=30,
                                  background_color="rgba(255, 255, 255, 0)",
                                  mode="RGBA").generate_from_frequencies(common_text_dict)
            # plt.imshow(wordcloud, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()
            # if len(selected_id_lst) == 1:
            #     wordcloud.to_file(
            #         os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png"))
            #     print("success")
            #
            #     self.label.setPixmap(
            #         QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png")))
            #     self.label.setScaledContents(True)
            # else:
            #     wordcloud.to_file(
            #         os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png"))
            #
            #     self.label.setPixmap(
            #         QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png")))
            #     self.label.setScaledContents(True)

            wordcloud.to_file(
                os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png"))

            self.label.setPixmap(
                QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png")))
            self.label.setScaledContents(True)

            # # text = []
            # selected_count_df = []
            # for i in range(len(selected_id_lst)):
            #     text = []
            #     # 某一篇的文章内容
            #     text_content = self.all_pro_data[self.all_pro_data['id'].isin(selected_id_lst)].iloc[i]["content"]
            #     text.append(text_content)
            #     count_matrix = coun_vect.fit_transform(text)
            #     count_array = count_matrix.toarray()
            #     # selected_count_array.append(count_array)
            #     df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
            #     print(df)
            #     text_dict = df.iloc[0].sort_values(ascending=False)[:80]
            #     print(type(text_dict))
            #     print(text_dict)
            #     # data = count_array, columns = coun_vect.get_feature_names_out()
            #     text_dict_df = pd.DataFrame({'words': text_dict.index, 'frequency': text_dict.values})
            #     selected_count_df.append(text_dict_df)
            #
            #     # print(type(count_array))
            #     # print(text_dict_df)
            #     # print("-"*20)
            #
            # # print(selected_count_df)
            # inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')),
            #                          selected_count_df)
            # # print(inter_text_dict.iloc[:, 1:])
            # inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
            # # inter_text_dict.loc['frequency'] = inter_text_dict.iloc[:, 1:].sum(axis=1)
            # # inter_text_dict = inter_text_dict.iloc[:1, :]
            # common_freq = inter_text_dict[["words", "frequency_total"]]
            # # print(inter_text_dict.iloc[:, :2])
            # print(common_freq)
            # # print("-"*10)
            # # print(common_freq.iloc[:]["words"])
            # # print(type(common_freq.iloc[:]["words"]))
            # # print("-"*10)
            # common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])
            #
            # common_text_dict = common_freq_series.sort_values(ascending=False)[:80]
            # print(common_text_dict)
            # common_text_dict = common_text_dict[common_text_dict > 3]
            #
            # wordcloud = WordCloud(width=500,
            #                       height=500,
            #                       max_words=50,
            #                       min_word_length=3,
            #                       prefer_horizontal=0.7,
            #                       scale=30,
            #                       background_color="rgba(255, 255, 255, 0)",
            #                       mode="RGBA").generate_from_frequencies(common_text_dict)
            # # plt.imshow(wordcloud, interpolation='bilinear')
            # # plt.axis("off")
            # # plt.show()
            # if len(selected_id_lst) == 1:
            #     wordcloud.to_file(
            #         os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png"))
            #     print("success")
            #
            #     self.label.setPixmap(
            #         QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png")))
            #     self.label.setScaledContents(True)
            # else:
            #     wordcloud.to_file(
            #         os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png"))
            #
            #     self.label.setPixmap(
            #         QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png")))
            #     self.label.setScaledContents(True)

        # text_temp = self.all_data[self.all_data['id'].isin(selected_id_list)].iloc[0]["content"]
        # # print(selected_row_id)
        # print(text_temp)
        #
        # text = []
        # text.append(text_temp)
        #
        #
        # coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        # count_matrix = coun_vect.fit_transform(text)
        # count_array = count_matrix.toarray()
        # df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
        # # print(df.iloc[0])
        #
        # text_dict = df.iloc[0].sort_values(ascending=False)[:50]
        # text_dict = text_dict[text_dict > 3]
        # # print(text_dict)
        # wordcloud = WordCloud(width=500,
        #                       height=500,
        #                       max_words=50,
        #                       min_word_length=3,
        #                       scale=25,
        #                       background_color="rgba(255, 255, 255, 0)",
        #                       mode="RGBA").generate_from_frequencies(text_dict)
        # #plt.imshow(wordcloud, interpolation='bilinear')
        # #plt.axis("off")
        # #plt.show()
        # wordcloud.to_file(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + selected_row_id + ".png"))
        #
        # self.label.setPixmap(QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + selected_row_id + ".png")))
        # self.label.setScaledContents(True)

    def dataHead(self):
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)

        publication = self.comboBox_Pub.currentText()
        year = self.comboBox_Year.currentText()

        if publication == "All":
            if year == "All":
                self.displayed_data = self.all_data.iloc[:, 1:3]
                print(self.displayed_data)
                # print("all, all")
            else:
                self.displayed_data = self.all_data.loc[self.all_data["year"] == int(year), ["id", "title"]]
                print(self.displayed_data)
                # print("all, no-all")
        else:
            if year == "All":
                self.displayed_data = self.all_data.loc[self.all_data["publication"] == publication, ["id", "title"]]
                print(self.displayed_data)
                # print("no-all, all")
            else:
                self.displayed_data = self.all_data.loc[
                    (self.all_data["publication"] == publication) & (self.all_data["year"] == int(year)), ["id",
                                                                                                           "title"]]
                print(self.displayed_data)
                # print("no-all, no-all")

        NumRows = len(self.displayed_data.index)

        self.tableWidget.setColumnCount(len(self.displayed_data.columns))
        self.tableWidget.setRowCount(NumRows)
        self.tableWidget.setHorizontalHeaderLabels(self.displayed_data.columns)
        self.tableWidget.horizontalHeader().setStyleSheet("QHeaderView { font-size: 15pt; "
                                                          "background-color:rgb(240, 248, 255);"
                                                          "font-weight: bold;}")
        self.tableWidget.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        self.tableWidget.setStyleSheet("alternate-background-color: rgb(220, 220, 220);"
                                       "background-color: rgb(245, 245, 245);")

        for i in range(NumRows):
            for j in range(len(self.displayed_data.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.displayed_data.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

        # self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
        #
        # NumRows = len(self.all_data.index)
        #
        # self.tableWidget.setColumnCount(len(self.all_data.columns) - 2)
        # self.tableWidget.setRowCount(NumRows)
        # self.tableWidget.setHorizontalHeaderLabels(self.all_data.columns)
        #
        # for i in range(NumRows):
        #     for j in range(len(self.all_data.columns)-2):
        #         self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_data.iat[i, j])))
        #
        # self.tableWidget.resizeColumnsToContents()
        # self.tableWidget.resizeRowsToContents()

    def detailed_data(self):

        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)

        # Clear textEdit
        self.textEdit.clear()

        arti_id = self.tableWidget.selectedItems()[0].text()  # 获取选中文本所在的行
        # print(arti_id)
        # print(type(row))

        self.selected_data = self.all_data[self.all_data['id'].isin([int(arti_id)])]
        # print(self.searched_data)

        self.textEdit.setPlainText(
            "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
            .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3],
                    self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6],
                    self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

        # print(
        #     "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
        #     .format(self.searched_data.iloc[0, 1], self.searched_data.iloc[0, 2], self.searched_data.iloc[0, 3],
        #             self.searched_data.iloc[0, 4], self.searched_data.iloc[0, 5], self.searched_data.iloc[0, 6],
        #             self.searched_data.iloc[0, 7], self.searched_data.iloc[0, 9]))

        # column = self.tableWidget.selectedItems()[0].column()  # 获取选中文本所在的列

        # print("Id = {0}\n News Title = {1}\n Publication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nContent = {6}" +
        #       "Month = {6}\n" + "content = {7}\n".format(self.all_data.iloc[row, 0],
        #       self.all_data.iloc[row, 1], self.all_data.iloc[row, 2], self.all_data.iloc[row, 3],
        #       self.all_data.iloc[row, 4], self.all_data.iloc[row, 5], self.all_data.iloc[row, 6], self.all_data.iloc[row, 8]))

        # #####
        # self.textEdit.setPlainText("Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
        #       .format(self.all_data.iloc[row, 1], self.all_data.iloc[row, 2], self.all_data.iloc[row, 3], self.all_data.iloc[row, 4], self.all_data.iloc[row, 5], self.all_data.iloc[row, 6], self.all_data.iloc[row, 7], self.all_data.iloc[row, 9]))
        # #####

        # for i in range(len(self.all_columns)):
        #     contents = self.df.iloc[row, i]  # 获取选中文本内容
        #     print('id = {0} - title = {1}'.format(X['id'].loc[i], X['title'].loc[i]))
        #     print("Id = {0}\n"+"News Title = {1}\n"+"Publication = {2}\n"+
        #           "Author = {3}\n"+"Date = {4}\n"+"Year = {5}\n"+
        #           "Month = {6}\n"+"content = {7}\n".format(self.df.iloc[row, 0]),self.df.iloc[row, 1]))
        #

        # print("选择的内容为：", contents)
        # print("所选的内容所在的行为：", row)
        # print("所选的内容所在的列为：", column)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sheet = MainWindow()
    sheet.show()
    sys.exit(app.exec_())

    # app = QApplication(sys.argv)
    # ex = Ui_Form()
    # w = QWidget()
    # ex.setupUi(w)
    # w.show()
    # sys.exit(app.exec_())
