import os
import tkinter as tk
from App.take_photos_predict import *


if __name__ == '__main__':

 class CustomMessageBox(tk.Toplevel):

     def __init__(self, parent, text):
         tk.Toplevel.__init__(self, parent)
         tk.Label(self, text=text, font=('arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, padx=10, pady=80)

         button_yes = tk.Button(self, text="Yes", font=('arial', 11, 'bold'), fg="green", command=self.yes, width=8)
         button_yes.grid(row=1, column=0, padx=10, pady=10)
         button_no = tk.Button(self, text="No", font=('arial', 11, 'bold'), fg = "red", command=self.no, width=8)
         button_no.grid(row=1, column=1, padx=10, pady=10)

         self.answer = None
         self.protocol("WM_DELETE_WINDOW", self.no)

     def yes(self):
         self.answer = "Yes"
         self.destroy()

     def no(self):
         self.answer = "No"
         self.destroy()
     def no_response(self):
         self.destroy()


 def ask():
     ans = CustomMessageBox(root, "The system assumes you are sleeping. \n You have 10 seconds until shutdown. \n Would you like to continue using the app?")
     root.after(10000, ans.no_response)
     root.wait_window(ans)
     return ans.answer


 root = tk.Tk()
 root.configure(background='pale turquoise')
 root.title('Shut Down my PC When I Fall Asleep')

 canvas1 = tk.Canvas(root, width=300, height=300)
 canvas1.pack()



 def start():
     time_interval = interval.get()
     assume(time_interval)
     response = ask()
     if(response == "Yes"):
         start()
     elif(response == "No"):
         root.destroy()
     else:
         print("It will shut down now")
         os.system('shutdown /p /f')



 button1 = tk.Button(text='Start',font=('arial', 12, 'bold'), relief="raised",command=start, bg='turquoise3', fg='white', height = 5, width = 10)
 interval = tk.Scale(root, from_=0, to=600, length=600, tickinterval=60, activebackground='turquoise3', label="Please select a time interval in seconds: ", font=('arial', 10, 'bold'), fg='turquoise4', orient=tk.HORIZONTAL)
 interval.set(0)
 interval.pack()

 canvas1.create_window(150, 150, window=button1)

 root.mainloop()

