# dbconnection.py
import pymysql

def get_connection():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='Gurbir@06',
        database='mydb'
    )
    return conn