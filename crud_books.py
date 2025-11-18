import mysql.connector

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Gurbir@06", 
    database="mydb"
)

with connection.cursor() as cursor:

    # Insert a record
    insert_query = "INSERT INTO books_db (book_name) VALUES (%s)"
    value = ("Python Programming",)
    cursor.execute(insert_query, value)
    connection.commit()
    print(" Record inserted successfully!")

    # Alter table
    alter_query = "ALTER TABLE books_db ADD COLUMN author VARCHAR(200)"
    try:
        cursor.execute(alter_query)
        connection.commit()
        print(" Table altered successfully — 'author' column added!")
    except:
        print(" Column 'author' may already exist, skipping...")

    # Update a record
    update_query = "UPDATE books_db SET author = %s WHERE book_id = %s"
    values = ("Guido van Rossum", 1)
    cursor.execute(update_query, values)
    connection.commit()
    print(" Record updated successfully!")

    # Delete a record
    delete_query = "DELETE FROM books_db WHERE book_id = %s"
    value = (2,)
    cursor.execute(delete_query, value)
    connection.commit()
    print(" Record deleted successfully!")

connection.close()
print(" All operations completed successfully!")
