import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os


from pipeline import read_meter

OUTPUT_FOLDER = "output/crops"
EXCEL_PATH = "output/results.xlsx"


class MeterApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Electric Meter Reader AI")
        self.root.geometry("600x450")

        self.image_paths = []
        self.results = []

        tk.Label(
            root,
            text="Electric Meter AI",
            font=("Arial", 18,"bold")
        ).pack(pady=10)

        # Buttons frame
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(
            btn_frame,
            text="Add Images",
            command=self.add_images,
            width=18
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            btn_frame,
            text="Remove Selected",
            command=self.remove_selected,
            width=18
        ).grid(row=0, column=1, padx=5)

        tk.Button(
            btn_frame,
            text="Reset",
            command=self.reset_app,
            width=18,
            bg="red",
            fg="white"
        ).grid(row=0, column=2, padx=5)

        # Listbox
        self.listbox = tk.Listbox(root, height=10)
        self.listbox.pack(fill="both", padx=10, pady=10)

        # Process button
        tk.Button(
            root,
            text="Process Images",
            command=self.process_images,
            height=2,
            width=30
        ).pack(pady=5)

        # Export button
        tk.Button(
            root,
            text="Export Excel",
            command=self.export_excel,
            height=2,
            width=30
        ).pack(pady=5)

        # Log window
        self.log = tk.Text(root, height=15)
        self.log.pack(fill="both", padx=10, pady=10)

    # -----------------------------
    def write_log(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    # -----------------------------
    def add_images(self):

        files = filedialog.askopenfilenames(
            title="Select Meter Images",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )

        for file in files:
            if file not in self.image_paths:
                self.image_paths.append(file)
                self.listbox.insert(tk.END, file)

    # -----------------------------
    def remove_selected(self):

        selected = self.listbox.curselection()

        for index in reversed(selected):
            self.listbox.delete(index)
            del self.image_paths[index]

    # -----------------------------
    # ⭐ FIXED FUNCTION
    # -----------------------------
    def process_images(self):

        if not self.image_paths:
            messagebox.showwarning("Warning", "No images selected!")
            return

        self.results.clear()

        for img_path in self.image_paths:

            filename = os.path.basename(img_path)
            self.write_log(f"Processing {filename}...")

            result = read_meter(img_path, OUTPUT_FOLDER)

            if result is not None:

                house_number, meter_number = result

                self.results.append([
                    filename,
                    house_number,
                    meter_number
                ])

                self.write_log(
                    f"House: {house_number} | Meter: {meter_number}"
                )

            else:
                self.results.append([
                    filename,
                    "NOT FOUND",
                    "NOT FOUND"
                ])

                self.write_log("No meter detected")

        messagebox.showinfo("Done", "Processing Finished!")

        # -----------------------------
    def reset_app(self):

        confirm = messagebox.askyesno(
            "Reset",
            "Clear all images and results?"
        )

        if not confirm:
            return

        # clear stored data
        self.image_paths.clear()
        self.results.clear()

        # clear listbox
        self.listbox.delete(0, tk.END)

        # clear log window
        self.log.delete("1.0", tk.END)

        self.write_log("Application reset complete.")

    # -----------------------------
    # ⭐ FIXED EXPORT


# --------------------------------
    def export_excel(self):

        if not self.results:
            messagebox.showwarning("No Data", "No results to export.")
            return

        # Create dataframe
        df = pd.DataFrame(self.results)

        # Ask user where to save
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel File", "*.xlsx")],
            title="Save Excel File As"
        )

        # User pressed cancel
        if not file_path:
            return

        # Save file
        df.to_excel(file_path, index=False)

        self.write_log(f"Excel exported to:\n{file_path}")
        messagebox.showinfo("Success", "Excel file saved successfully!")

# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MeterApp(root)
    root.mainloop()