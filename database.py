import sqlite3

conn = sqlite3.connect('picturesdatabase.db')
cursor = conn.cursor()

cursor.execute('''
  CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    ad_name TEXT,
    upload_date TEXT,
    filename TEXT,
    female INTEGER,
    male INTEGER,
    child INTEGER,
    young_adult INTEGER,
    middle_aged INTEGER,
    elderly INTEGER,  
    happy INTEGER,
    neutral INTEGER,
    sad INTEGER 
  )
''')



# Adatok hozzáadása a users táblához
# cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('user1', 'user1@example.com'))

# Lekérdezés az images táblából
cursor.execute("SELECT * FROM images")
# result = cursor.fetchall()

all_records = cursor.fetchall()

for record in all_records:
    print(record)



conn.commit()  # Az adatok mentése
conn.close()   # Az adatbázis bezárása