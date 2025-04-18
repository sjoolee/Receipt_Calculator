import requests
import pandas as pd
from collections import defaultdict
from API_project_4TN4 import extract_prices, comparison
from blur_detection import user_images

##############################
# USER CONTROLS #
# Import image
img = user_images()

# API
api_url = 'https://api.api-ninjas.com/v1/imagetotext'
image_file_descriptor = open(img, 'rb')
files = {'image': image_file_descriptor}
r = requests.post(api_url, files=files, headers={'X-Api-Key': 'YOUR_API_KEY'})

response_data = r.json()

# Extract prices from OCR response
receipt_data = extract_prices(response_data)

print("\n📄 Initial Receipt Data:")
df = pd.DataFrame(receipt_data.items(), columns=['Item', 'Price'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Convert prices to numeric values
print(df)

## Table Modifications
while True:
    print("\n📌 Receipt Modifications:")
    print("1 - Edit an item name")
    print("2 - Edit a price")
    print("3 - Delete a row")
    print("4 - Finish and proceed")
    
    choice = input("Select an option (1/2/3/4): ").strip()

    if choice == '1':  # Edit an item name
        print("\nCurrent Data:")
        print(df)
        try:
            index = int(input("Enter the row index to edit the item name: "))
            new_item = input("Enter the new item name: ").strip()
            df.at[index, 'Item'] = new_item
            print("\n✅ Updated Data:")
            print(df)
        except (ValueError, IndexError):
            print("❌ Invalid input. Please enter a valid row index.")

    elif choice == '2':  # Edit a price
        print("\nCurrent Data:")
        print(df)
        try:
            index = int(input("Enter the row index to edit the price: "))
            new_price = float(input("Enter the new price: "))
            df.at[index, 'Price'] = new_price
            print("\n✅ Updated Data:")
            print(df)
        except (ValueError, IndexError):
            print("❌ Invalid input. Please enter a valid row index and numeric price.")

    elif choice == '3':  # Delete a row
        print("\nCurrent Data:")
        print(df)
        try:
            index = int(input("Enter the row index to delete: "))
            df = df.drop(index).reset_index(drop=True)
            print("\n✅ Updated Data:")
            print(df)
        except (ValueError, KeyError):
            print("❌ Invalid input. Please enter a valid row index.")

    elif choice == '4':  # Finish editing
        break

    else:
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

print("\n📄 Final Receipt Data:")
print(df)

# Ensure at least one item is selected
person_totals = defaultdict(float)
person_orders = defaultdict(list)  # Stores items each person ordered
item_splits = defaultdict(list)  # Stores who is splitting each item
items_selected = False  # Flag to check if anything was ordered

while not items_selected:
    # Reset totals for each retry
    person_totals.clear()
    person_orders.clear()

    while True:
        user_name = input("\n👤 Enter your name (or type 'done' if all users are entered): ").strip()
        if user_name.lower() == 'done':
            break

        print(f"\n🛒 {user_name}, select the items you ordered (enter row index). Type 'done' when finished.")
        print(df.to_string(index=True))

        while True:
            user_input = input("Enter item index: ").strip()
            if user_input.lower() == 'done':
                break
            try:
                index = int(user_input)
                item_name = df.at[index, 'Item']
                item_price = df.at[index, 'Price']

                # Display existing splits for the item
                if item_name in item_splits:
                    print(f"\n🔍 **{item_name}** has already been ordered by: {', '.join(item_splits[item_name])}")

                # Ask how many people are splitting this item
                shared_with = input(f"👥 Who are you sharing '{item_name}' with? (Enter names separated by commas, or 'none' if alone): ").strip()
                shared_people = [user_name] if shared_with.lower() == 'none' else [user_name] + [p.strip() for p in shared_with.split(",")]

                num_people = len(shared_people)
                per_person_price = item_price / num_people

                # Distribute cost among people & track items ordered
                for person in shared_people:
                    person_totals[person] += per_person_price
                    person_orders[person].append(f"{item_name} (${per_person_price:.2f})")

                # Track who is splitting the item
                item_splits[item_name].extend(shared_people)

                items_selected = True  # At least one item has been ordered

                # Print current splits for the item
                print(f"\n👥 {item_name} is now being split by: {', '.join(item_splits[item_name])}")

            except (ValueError, IndexError):
                print("❌ Invalid input. Please enter a valid row index.")

    if not items_selected:
        print("\n⚠️ **At least one item must be selected! Please enter at least one order.**")

# Proceed with calculations
subtotal = sum(person_totals.values())
tax_rate = 0.13  # Example: 13% tax
total_tax = subtotal * tax_rate

# Tip selection
print("\n💵 Tip Options:")
print("1 - 10%")
print("2 - 15%")
print("3 - 20%")
print("4 - Enter a custom tip amount")

tip_choice = input("Select an option (1/2/3/4): ").strip()

if tip_choice == '1':
    tip_percentage = 0.10
elif tip_choice == '2':
    tip_percentage = 0.15
elif tip_choice == '3':
    tip_percentage = 0.20
elif tip_choice == '4':
    while True:
        try:
            tip_percentage = float(input("Enter your custom tip percentage (e.g., 0.15 for 15%): "))
            break
        except ValueError:
            print("❌ Invalid input. Enter a numeric value.")
else:
    print("❌ Invalid choice. Defaulting to 15% tip.")
    tip_percentage = 0.15

total_tip = subtotal * tip_percentage
total_amount_due = subtotal + total_tax + total_tip

# Calculate per-person final total
for person in person_totals:
    person_totals[person] += (total_tax + total_tip) * (person_totals[person] / subtotal)

# Display final bill summary
print("\n💳 **Final Bill Summary**")
print(f"\nSubtotal: ${subtotal:.2f}")
print(f"Tax (13%): ${total_tax:.2f}")
print(f"Tip ({tip_percentage * 100}%): ${total_tip:.2f}")
print(f"Total Amount Due: ${total_amount_due:.2f}\n")

# Print individual breakdown
for person, amount in person_totals.items():
    print(f"\n💰 **{person} Pays: ${amount:.2f}**")
    print("📝 Ordered:")
    for item in person_orders[person]:
        print(f"   - {item}")

# Payment confirmation
confirm = input("\nConfirm payment? (yes/no): ").strip().lower()
if confirm == 'yes':
    print(f"\n🎉 **Payment Successful! Thank you!** 🎉")
else:
    print("\n❌ Payment canceled.")
