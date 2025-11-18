import pymysql

connection = pymysql.connect(
    host='localhost',
    user ='root',
    password='12345678',
    database='mydb'
)

with connection.cursor() as cursor:
        # insert data
        insert_query = "INSERT INTO  employee_data(emp_name,emp_role)VALUES(%s,%s)"
        values=("JHON","Software engineer")
        cursor.execute(insert_query,values)
        connection.commit()

        # read
        select_query ="SELECT * FROM employee_data"
        cursor.execute(select_query)
        rows=cursor.fetchall()
        print("\n employes list:")
        for row in rows:
               print(row)

        #update
        update_query="UPDATE employee_data set emp_role=%s WHERE emp_id = %s"
        cursor.execute(update_query,("Software engineer",1))
        connection.commit()
        print("\n employee salary updated!")


        #read 
        select_query ="SELECT * FROM employee_data"
        cursor.execute(select_query)
        rows=cursor.fetchall()
        print("\n employes list:")
        for row in rows:
               print(row)


        #delete 
        delete_query="DELETE FROM employee_data WHERE emp_id =%s"
        cursor.execute(delete_query,(1,))
        connection.commit()
        print("\n employee record deleted!")