import sqlite3
import csv

def export_sqlite_to_csv(database_file, table_name, csv_file):
    # Connect to SQLite database
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Execute SQL query to select all data from the specified table
    query = f"SELECT * FROM {table_name};"
    cursor.execute(query)

    # Fetch all the rows
    rows = cursor.fetchall()

    # Get the column names
    column_names = [description[0] for description in cursor.description]

    # Write to CSV file
    with open(csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow(column_names)
        
        # Write the data
        csv_writer.writerows(rows)

    # Close the connections
    cursor.close()
    conn.close()

# Example usage
database_file = 'instance/database_cb.db'
table_name = 'chat'
csv_file = 'output2_sum.csv'

export_sqlite_to_csv(database_file, table_name, csv_file)
