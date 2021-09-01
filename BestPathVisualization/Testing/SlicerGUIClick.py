import pyautogui
import time
import os

# Margin 5
# Needle 130
# only Max Val

mouse_console = (401,240)

button_dataLoad = (1345,105)
button_chosseFile = (1146,513)
button_open = (1563,838)
mouse_centerScreen = (1352,702)
button_ok = (1550,875)

button_module = (1874,99)
button_outputModel = (1604,325)
button_newModel = (1583,364)
button_overlaytype = (1617,516)
button_overlayAll = (1263,538)

field_fileName = (1588,572)
button_calcOverlay = (1530,593)
button_target = (1568,543)

button_closeSlicer = (2546,46)
button_closeSlicerConfirm = (1340,709)



def openSlicer():
    pyautogui.click(mouse_console) 
    pyautogui.typewrite('.')
    pyautogui.hotkey('shift', '7')
    pyautogui.typewrite('Slicer')
    pyautogui.press('enter') 

def loadData(str):
    pyautogui.click(button_dataLoad) 
    time.sleep(1)
    pyautogui.click(button_chosseFile) 
    time.sleep(1)
    for f in str:
        if f=='/':
            pyautogui.hotkey('shift', '7')
        else:
            pyautogui.typewrite(f)
    time.sleep(1)
    pyautogui.press('enter') 
    time.sleep(1)
    pyautogui.click(mouse_centerScreen) 
    pyautogui.click(mouse_centerScreen) 
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(1)
    pyautogui.click(button_open) 
    time.sleep(1)
    pyautogui.click(button_ok) 
    time.sleep(5)

def setupSlicer():
    pyautogui.click(button_module) 
    time.sleep(1)
    pyautogui.click(button_outputModel) 
    time.sleep(1)
    pyautogui.click(button_newModel) 
    time.sleep(1)
    pyautogui.click(button_overlaytype) 
    time.sleep(1)
    pyautogui.click(button_overlayAll) 
    time.sleep(1)
    
def waitUntilBlue():
   waiting = True
   while waiting:
       im = pyautogui.screenshot()
       val = im.getpixel((1776,648))
       print(val)
       if val[2] < 255:
           waiting = False
       else:
           time.sleep(0.1)
   
def iterateSlicer(fileName,numTargets):
    pyautogui.click(field_fileName) 
    time.sleep(1)
    for f in fileName:
        if f=='/':
            pyautogui.hotkey('shift', '7')
        else:
            pyautogui.typewrite(f)
    pyautogui.click(button_calcOverlay) 
    waitUntilBlue()      
    for p in range(numTargets-1):
        print((p))
        time.sleep(1)  
        pyautogui.click(button_target)
        pyautogui.press('down') 
        pyautogui.press('enter') 
        time.sleep(1) 
        pyautogui.click(button_calcOverlay) 
        time.sleep(2)
        waitUntilBlue()
    
def closeSlicer():
    pyautogui.click(mouse_console) 
    pyautogui.hotkey('ctrl', 'c')
    
    
    
files_to_do = ["/home/Neidhardt/Dokumente/Paper_Neidhardt_Gerlach/Annotated_CTs_II/Leiche_1/Bauch_30_03_2021/","/home/Neidhardt/Dokumente/Paper_Neidhardt_Gerlach/Annotated_CTs_II/Leiche_15/Ruecken_03_03_2021/"]

targets_to_do = [9,16]


for f in range(0,len(targets_to_do)):
    openSlicer()
    time.sleep(8)
    loadData(files_to_do[f])

    setupSlicer()
    files_out = files_to_do[f]
    files_out = files_out.replace("Annotated_CTs_II","Colormaps")
    if not os.path.exists(files_out):
        print("making path")
        os.makedirs(files_out)
    print(files_out)
    iterateSlicer(files_out,targets_to_do[f])
    closeSlicer()




