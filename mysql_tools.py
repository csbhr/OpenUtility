import pymysql


class MysqlBase():

    def __init__(self, host, port, user, passwd, db):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db

    # 获取游标
    def get_cursor(self):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db)
        self.cursor = self.conn.cursor()
        return self.cursor

    # 关闭连接
    def close_all(self):
        self.cursor.close()
        self.conn.close()

    # 执行更新语句sql
    def exec_update_sql(self, sql):
        cursor = self.get_cursor()
        try:
            count = cursor.execute(sql)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print("Skip one!  : " + str(e))
        self.close_all()
        return count

    # 执行一串更新语句sql_list
    def exec_update_sqlList(self, sql_list):
        cursor = self.get_cursor()
        count = 0
        for sql in sql_list:
            try:
                count = count + cursor.execute(sql)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                print("Skip one!  : " + str(e))
        self.close_all()
        return count

    # 执行查询语句sql
    def exec_query_sql(self, sql):
        cursor = self.get_cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        self.close_all()
        return rows
