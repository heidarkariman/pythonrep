import csv
import random

# Open the original CSV file for reading
with open('customer_data.csv', 'r') as original_file:
    reader = csv.reader(original_file)

    # Open a new CSV file for writing
    with open('new_customer_data.csv', 'w', newline='') as new_file:
        writer = csv.writer(new_file)

        # Loop through each row of the original CSV file and add a new "conversation" field with a randomly generated 0 or 1 value
        for i, row in enumerate(reader):
            if i == 0: # First row (header) - add the new field name
                row.append('conversation')
            else: # All other rows - add a randomly generated 0 or 1 value
                row.append(random.randint(0, 1))
            writer.writerow(row)

print('Done')