#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import time

import xlrd
import csv
import sys
from datetime import datetime
from utils import mkdir_p, flock, copy_file #, frelease, move_file

ARCHIVE_FOLDER = "Archive"
DATA_FOLDER = "Data"

def csv_from_excel(excel_file):
    workbook = xlrd.open_workbook(excel_file)
    filename = os.path.splitext(os.path.basename(excel_file))[0]
    date_str = filename.split("_")[-1]
    try:
        timestamp = datetime.strptime(date_str, '%Y%b%d')
    except:
        timestamp = datetime.strptime(date_str, '%Y%m%d')
    fl_path = os.path.join(os.path.dirname(excel_file), DATA_FOLDER, date_str)
    mkdir_p(fl_path)   
    
    all_worksheets = workbook.sheet_names()
    for worksheet_name in all_worksheets:
        worksheet = workbook.sheet_by_name(worksheet_name)
        if not worksheet_name.startswith('tab'):
            worksheet_name = 'tab' + worksheet_name
        worksheet_full_filepath = os.path.join(fl_path, ''.join([filename,'_',worksheet_name,'.csv']))
        if os.path.exists(worksheet_full_filepath):
            print("File duplicate [" + str(excel_file) + "] already export this file [" + worksheet_full_filepath + "]" )
            return

        your_csv_file = open(worksheet_full_filepath, 'wb')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
        
        header = True
        for rownum in range(0,worksheet.nrows):
            if header:
                header = False
                row = []
                row_len = 0
                for entry in worksheet.row_values(rownum):
                    data = unicode(entry).encode("utf-8")
                    data = data.replace("(s)","").replace(" [Y/N]","").replace("(","").replace(")","")
                    data = data.replace(" ","_").replace(".","_").replace("/","p")
                    data = data.replace("#","Count").replace("%","Percent").replace("Host","fHost")
                    row.append(data)
                    row_len += len(data)
                row.insert(0,'timestamp')

                if row_len == 0:
                    header = True
                    continue
            else:
                # row = [unicode(entry).encode("utf-8") for entry in worksheet.row_values(rownum)]
                # row.insert(0,str(timestamp))
                # print(row)
                actual_values = []
                for c in range(0, worksheet.ncols):
                    
                    cell = worksheet.cell(rownum, c)
                    actual = cell.value

                    if cell.ctype == xlrd.XL_CELL_DATE:
                        actual = datetime(*xlrd.xldate_as_tuple(actual, worksheet.book.datemode))
                        actual = actual.strftime("%d/%m/%Y %H:%M:%S")
                    actual_values.append(actual)
                    # print(actual_values)
                row = actual_values
                row.insert(0,str(timestamp))
            wr.writerow(row)
        your_csv_file.close()

if __name__ == "__main__":
    # csv_from_excel(sys.argv[1])
    # data_file = os.path.abspath(sys.argv[1])
    data_file = "C:\\Users\\suesa\\Desktop\\splitter\\RVTools_testserver_20191219.xlsx"
    
    archive_file = os.path.join(os.path.dirname(data_file), ARCHIVE_FOLDER, os.path.basename(data_file))
    flock(data_file)
    if os.path.exists(archive_file):
        print("File duplicate [" + str(data_file) + "] already have this file in Archive [" + archive_file + "]")
    else:
        csv_from_excel(data_file)
        # frelease(data_file)
        copy_file(data_file, os.path.join(os.path.dirname(data_file), ARCHIVE_FOLDER))

