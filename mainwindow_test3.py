from functools import reduce

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox, \
    QAbstractItemView, QVBoxLayout
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIntValidator
import pandas as pd
import numpy as np
import sys
from untitled3 import Ui_Form
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
# from wcwidget import WcWidget


# class MainWindow(QWidget, Ui_Form):
class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        # super(MainWindow, self).__init__(parent=parent)
        # ui = Ui_Form()
        # ui.setupUi(self)

        # super(MainWindow, self).__init__(parent=parent)
        # self.setupUi(self)

        super(MainWindow, self).__init__()
        QWidget.__init__(self)
        self.setupUi(self)

        self.lineEdit.setPlaceholderText(" keywords")
        self.lineEdit_2.setPlaceholderText(" id: 17283~73469")
        # 整数校验器[1, 99] 精度小数点后两位
        intvalidator = QIntValidator()
        intvalidator.setRange(17283, 73469)
        self.lineEdit_2.setValidator(intvalidator)

        #加载数据
        self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
        self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
        self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

        self.all_text_content = self.all_data['content']
        self.vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True, stop_words='english',
                                 max_df=0.3)
        self.vector_fit = self.vector.fit_transform(self.all_text_content)

        #self.ButtonOpen.clicked.connect(self.OpenFile)
        self.pushButton.clicked.connect(self.dataHead)
        self.pushButton_2.clicked.connect(self.indi_wordcloud)
        self.pushButton_3.clicked.connect(self.search_articles)
        self.pushButton_4.clicked.connect(self.find_similar_articles)
        self.pushButton_5.clicked.connect(self.multi_wordcloud)

        self.pushButton_6.clicked.connect(self.update_graph)

        self.addToolBar(NavigationToolbar(self.WcWidget.canvas, self))

        self.tableWidget.itemDoubleClicked.connect(self.detailed_data)

    def update_graph(self):

        fs = 500
        f = random.randint(1, 100)
        ts = 1 / fs
        length_of_signal = 100
        t = np.linspace(0, 1, length_of_signal)

        cosinus_signal = np.cos(2 * np.pi * f * t)
        sinus_signal = np.sin(2 * np.pi * f * t)

        self.WcWidget.canvas.axes.clear()
        self.WcWidget.canvas.axes.plot(t, cosinus_signal)
        self.WcWidget.canvas.axes.plot(t, sinus_signal)
        self.WcWidget.canvas.axes.legend(('cosinus', 'sinus'), loc='upper right')
        self.WcWidget.canvas.axes.set_title('Cosinus - Sinus Signal')
        self.WcWidget.canvas.draw()




    def multi_wordcloud(self):

        # fig = plt.figure()

        # selected_id_lst = [53775, 53892]
        selected_id_lst = []
        for idx in self.tableWidget.selectionModel().selectedRows():
            selected_id_lst.append(int(self.tableWidget.item(idx.row(), 0).text()))

        coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        # text = []
        selected_count_df = []
        for i in range(len(selected_id_lst)):
            text = []
            # 某一篇的文章内容
            text_content = self.all_data[self.all_data['id'].isin(selected_id_lst)].iloc[i]["content"]
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

        # text1_fre_df = selected_count_df[0]
        # text2_fre_df = selected_count_df[1]
        # # print(text1_fre_df)
        # t1_fre_df_wd = pd.Series(text1_fre_df['frequency'].values, index=text1_fre_df.iloc[:]["words"])
        # t1_fre_df_wd = t1_fre_df_wd.sort_values(ascending=False)[:80]
        # # print(t1_fre_df_wd)
        # t2_fre_df_wd = pd.Series(text2_fre_df['frequency'].values, index=text2_fre_df.iloc[:]["words"])
        # t2_fre_df_wd = t2_fre_df_wd.sort_values(ascending=False)[:80]
        # # print("-"*30)
        # # print(text2_fre_df)
        # # print("-"*30)

        inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')),
                                 selected_count_df)
        # print(inter_text_dict.iloc[:, 1:])

        inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
        # print(inter_text_dict)

        # 获得共同词， 返回list
        inter_text_lst = np.array(inter_text_dict.iloc[:]["words"]).tolist()
        print(inter_text_lst)

        # 设置新列，确认是否为共同词，是 改为1
        for j in range(len(selected_id_lst)):
            text_x_is_in_common = [1 if selected_count_df[j].iloc[i]["words"] in inter_text_lst else 0 for i in
                                  range(selected_count_df[j].shape[0])]
            selected_count_df[j]["is_in_common"] = pd.DataFrame(text_x_is_in_common, columns=['is_in_common'])
            # text1_fre_df.groupby("is_in_common")["is_in_common"]类型为Series
            selected_count_df[j]["color"] = selected_count_df[j].groupby("is_in_common")["is_in_common"]. \
                transform(lambda x: "#000000" if (x.index).all() else "#00FFFF")

            txt_x_color_to_words = selected_count_df[j].groupby("color")["words"].agg(list).to_dict()

            txt_x_fre_df_wd = pd.Series(selected_count_df[j]['frequency'].values, index=selected_count_df[j].iloc[:]["words"])
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
            customed_color_func = Func_implement.SimpleGroupedColorFunc(txt_x_color_to_words, default_color)

            # Apply our color function
            wordcloud.recolor(color_func=customed_color_func)

            # plt.imshow(wordcloud, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()

            # ax.imshow(wordcloud)
            # ax.axis("off")


            wordcloud.to_file(
                                os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[j]) + ".png"))

        self.label.setText("word clouds saved sucessfully.")
    def find_similar_articles(self):
        self.tableWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        total_id = []
        for i in range(self.all_data.shape[0]):
            total_id.append(self.all_data.iloc[i]["id"])
        # total_id = all_data["id"]
        # print(total_id)
        requested_id = self.lineEdit_2.text()
        #requested_id_int = int(requested_id)
        if len(requested_id)==0:
            QMessageBox.critical(self, "error", "ID cannot be null.")
        elif int(requested_id) in total_id:
            # self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
            # self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
            # self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

            # self.all_text_content = self.all_data['content']
            # vector = TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True, stop_words='english',
            #                          max_df=0.3)
            # vector_fit = vector.fit_transform(self.all_text_content)

            # print(requested_id)
            # print(type(requested_id))
            index = self.all_data[self.all_data["id"] == int(requested_id)].index.tolist()[0]

            indices = Func_implement.find_similar(self.vector_fit, index)
            # print(indices)

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

            indices = Func_implement.search(self.vector_fit, self.vector, request)

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

            #text_content = self.all_data['content']


            #Func_implement.print_result(request, result, self.all_data)

            #self.textEdit.setPlainText(Func_implement.print_result(request, result, self.all_data))

            for i in range(NumRows):
                for j in range(len(self.searched_data.columns)):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.searched_data.iat[i, j])))

            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()


    def indi_wordcloud(self):


        selected_id_lst = []
        for idx in self.tableWidget.selectionModel().selectedRows():
            selected_id_lst.append(int(self.tableWidget.item(idx.row(), 0).text()))

        # print(selected_id_lst)
        if len(selected_id_lst) == 0:
            QMessageBox.critical(self, "error", "Please select article(s).")
        else:
            coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
            # text = []
            selected_count_df = []
            for i in range(len(selected_id_lst)):
                text = []
                # 某一篇的文章内容
                text_content = self.all_data[self.all_data['id'].isin(selected_id_lst)].iloc[i]["content"]
                text.append(text_content)
                count_matrix = coun_vect.fit_transform(text)
                count_array = count_matrix.toarray()
                # selected_count_array.append(count_array)
                df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
                text_dict = df.iloc[0].sort_values(ascending=False)[:80]
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
            if len(selected_id_lst) == 1:
                wordcloud.to_file(
                    os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png"))
                print("success")

                self.label.setPixmap(
                    QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/" + str(selected_id_lst[0]) + ".png")))
                self.label.setScaledContents(True)
            else:
                wordcloud.to_file(
                    os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png"))

                self.label.setPixmap(
                    QPixmap(os.path.join("/Users/chenkanyu/Desktop/arti/archive/words_in_common.png")))
                self.label.setScaledContents(True)


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
                self.displayed_data = self.all_data.loc[(self.all_data["publication"] == publication) & (self.all_data["year"] == int(year)), ["id", "title"]]
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

        arti_id = self.tableWidget.selectedItems()[0].text()  # 获取选中文本所在的行
        # print(type(row))

        self.selected_data = self.all_data[self.all_data['id'].isin([int(arti_id)])]
        #print(self.searched_data)

        self.textEdit.setPlainText("Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
              .format(self.selected_data.iloc[0, 1], self.selected_data.iloc[0, 2], self.selected_data.iloc[0, 3], self.selected_data.iloc[0, 4], self.selected_data.iloc[0, 5], self.selected_data.iloc[0, 6], self.selected_data.iloc[0, 7], self.selected_data.iloc[0, 9]))

        # print(
        #     "Id = {0} \nNews Title = {1} \nPublication = {2} \nAuthor = {3} \nDate = {4} \nYear = {5} \nMonth = {6} \nContent = {7}"
        #     .format(self.searched_data.iloc[0, 1], self.searched_data.iloc[0, 2], self.searched_data.iloc[0, 3],
        #             self.searched_data.iloc[0, 4], self.searched_data.iloc[0, 5], self.searched_data.iloc[0, 6],
        #             self.searched_data.iloc[0, 7], self.searched_data.iloc[0, 9]))

        #column = self.tableWidget.selectedItems()[0].column()  # 获取选中文本所在的列

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

        #print("选择的内容为：", contents)
        #print("所选的内容所在的行为：", row)
        #print("所选的内容所在的列为：", column)




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










