import heapq
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from tkinter import *
import ttkbootstrap as ttk
from PIL import Image, ImageTk

# Blue rectangle shows the runway strip. 
# The blue filled rectangle above the runway shows the parking/dispersal area. 
# On clicking on the runway, the unfilled red circle represents the CEP around the aimpoint. 
# The filled circle represents the strikepoint calculated based on the Gaussian probability density function.
# The unfilled green circle represents the submunition dispersal radius around the strikepoint.
# The green dots inside this circle represent the submunition strikepoints.
# The red dotted lines represent the MOS strips on the runway.

# For Parking Area
# The aforementioned conditions apply for the parking area as well with the parking area dimensions. 
# After clicking inside the parking area, the highlighted green circle show the area overlapping between the parking area and the area destroyed by the missile.






scale_length = 1 / 10  # e.g., 1 pixel = 10 feet for length
scale_width = 1 / 2  # e.g., 1 pixel = 2 feet for width
parking_scale_length = 1 / 10   # e.g., 1 pixel = 10 feet for length
parking_scale_width = 0.15  # e.g., 1 pixel = 6 feet for width
image = np.zeros((1220, 500, 3), dtype=np.uint8)
message='\n --------------------------------------------\n'
cep_values = []        # Contains the CEP values used to make the graph
up_down_point=[]        # Used in graph
rect_point=[]       # Contains topleft and bottom most point of rectangale
missile=[] 

def feet_to_pixels_length(feet, scale_length):   
    return int(feet * scale_length)

def feet_to_pixels_width(feet, scale_width):
    return int(feet * scale_width)

def runway_draw(length_ft,width_ft):    # draw runway (rectangle) for a given length and breadth 
    global image
    length_px = feet_to_pixels_length(length_ft, scale_length)
    width_px = feet_to_pixels_width(width_ft, scale_width)
    image_height = width_px + 235  # Add some padding for width
    image_width = length_px + 20 # Add some padding for length
    image = np.zeros((image_height+100, image_width+100, 3), dtype=np.uint8)
    top_left_corner = (50, 250)  # Adding 10-pixel padding
    rect_point.append(list(top_left_corner)) 
    bottom_right_corner = (top_left_corner[0] + length_px, top_left_corner[1] + width_px)
    rect_point.append(list(bottom_right_corner))
    cv2.rectangle(image, top_left_corner, bottom_right_corner, (255, 0, 0), 2)
    cv2.imshow("image", image)
    return rect_point

def parking_rectangle(length,breadth):  # draw parking (rectangle) for a given length and breadth 
    global image
    parking_rectangle_points=[]
    length_px = feet_to_pixels_length(length, parking_scale_length)
    width_px = feet_to_pixels_width(breadth, parking_scale_width)
    global parking_area
    parking_area = length_px*width_px
    parking_top_corner=(rect_point[0][0],rect_point[0][1]-width_px)
    parking_right_corner=(rect_point[0][0]+length_px,rect_point[0][1])
    parking_rectangle_points.append(parking_top_corner)
    parking_rectangle_points.append(parking_right_corner)
    cv2.rectangle(image,parking_top_corner,parking_right_corner,(100,0,0),-1)
    cv2.imshow("image", image)
    return parking_rectangle_points

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):  # To draw dotted line denoting MOS strips
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def generate_strike_ideal_point_new(rect_point,mos,subdisp):   #
       length = rect_point[1][0]-rect_point[0][0]
       width  = rect_point[1][1]-rect_point[0][1]
       cross_cen=[]
       cross_cen_x=[]
       cross_cen_y=[]
       lmos=mos[0]
       wmos=mos[1]
       total_strip_x=length//lmos
       total_strip_y=width//wmos
       total_strike_point=width//(subdisp*2)

       for xi in range(1,total_strip_x+1,1):
           drawline(image, (rect_point[0][0] +xi*lmos,rect_point[0][1]), (rect_point[0][0] +xi*lmos,rect_point[1][1]), (0, 0, 255), thickness=2, style='dotted', gap=20)
       for yi in range(1,total_strip_y+1,1):
           drawline(image, (rect_point[0][0],rect_point[0][1]+yi*wmos), (rect_point[1][0],rect_point[0][1]+yi*wmos), (0, 0, 255), thickness=2, style='dotted', gap=20 )
          
       for xi in range(1,total_strip_x+1,1):
           x1=int(rect_point[0][0]+xi*lmos-lmos/3)
           cross_cen_x.append((x1))

       for yi in range(1,total_strike_point+1,1):
           y1=int(rect_point[0][1]+yi*2*subdisp-(2*subdisp)/3)
           cross_cen_y.append((y1))

       for i in range(len(cross_cen_x)):
            for j in range(len(cross_cen_y)):
                 cross_cen.append([cross_cen_x[i],cross_cen_y[j]])

       for i in range (len(cross_cen)):
                 cv2.line(image, (cross_cen[i][0] - 10, cross_cen[i][1]), (cross_cen[i][0] + 10, cross_cen[i][1]), (0,0,255), 2)
                 cv2.line(image, (cross_cen[i][0], cross_cen[i][1] - 10), (cross_cen[i][0], cross_cen[i][1] + 10), (0,0,255), 2)

       cv2.imshow('image',image)
       return (total_strip_x,total_strip_y,total_strike_point,cross_cen)


def max_min_y(points): #returns maximum and minimum values in array 
    max=-10000000
    min=100000000
    for i in points:
        if (i[1]>max):
            max=i[1]
        if (i[1]<min):
             min=i[1]
    maxmin_y=(max,min)
    return maxmin_y

def generate_submunition_points(center, subdisp,submun): #generating submunition's strike points (green filled circle)
    global message
    points = []
    y=np.random.normal(85,5)
    total_successful_submunition=round(y/100*submun)
    r = np.random.uniform(0,subdisp,total_successful_submunition)
    theta = np.random.uniform(0,2*(np.pi),total_successful_submunition)
    for i in range(total_successful_submunition):
        points.append((int(center[0]+r[i]*np.cos(theta[i])),int(center[1]+r[i]*np.sin(theta[i]))))    
    message=str(total_successful_submunition)+" out of "+ str(submun)+ " submunitions have successfully hit \n "+str(submun-total_successful_submunition)+" have failed"    
    return points,total_successful_submunition

def generate_strike_point(aim_point, cep): # To generate strike point of missile
    list_y=[]
    list_x=[]
    x, y = aim_point
    dx,dy = np.random.normal(0, cep,2)
    list_x.append(dx)
    list_y.append(dy)
    strike_point = (x + dx, y + dy)
    return strike_point


def diff(arr): # To calculate length of strips remaining
     diff=[]
     b = rect_point[1][1]
     arr.append((b,b))
     arr.sort()
     for i in range(1,len(arr),1):
          diff1=arr[i][0]-arr[i-1][1]
          if(diff1<=0):
               diff.append(0)
          else:
               diff.append(diff1)
     text6="Lengths of the strips remaining: "+str(diff)+"\n"
     output_text.insert("end",text6)
     return diff

   
def on_mouse_click(event, x, y, flags, param): # functions for when user clicks
    if event == cv2.EVENT_LBUTTONDOWN:            
            if random.random()<0.85:
                global cep
                global subdisp
                points = []
                temp_message=''
                parking_rectangle_points
                if parking_rectangle_points[0][0]<x<parking_rectangle_points[1][0] and parking_rectangle_points[0][1]<y<parking_rectangle_points[1][1]:
                    cv2.circle(image,(x,y),parking_cep,(0,0,255),1)
                    strike_point=generate_strike_point((x,y),.6*parking_cep)
                    strike_point=np.round(strike_point).astype(int)
                    image2=image.copy()
                    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    rect_x, rect_y, rect_w, rect_h = parking_rectangle_points[0][0],parking_rectangle_points[0][1],parking_rectangle_points[1][0],parking_rectangle_points[1][1]
                    cv2.rectangle(image2, (rect_x, rect_y), (rect_w, rect_h), (0, 0,255), 2)
                    center = (int(strike_point[0]),int(strike_point[1]))
                    radius = parking_subdisp
                    cv2.circle(image, center, radius, (0, 0, 255), 2)
                    rect_mask = np.zeros_like(gray)
                    cv2.rectangle(rect_mask, (rect_x, rect_y), ( rect_w, rect_h), (255,0,255), -1)
                    circle_mask = np.zeros_like(gray)
                    cv2.circle(circle_mask, center, radius, 255, -1)
                    intersection = cv2.bitwise_and(rect_mask, circle_mask)
                    intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in intersection_contours:
                        cv2.drawContours(image, [contour], -1, (0,255,0), 2)

                    intersection_area = sum(cv2.contourArea(contour) for contour in intersection_contours)
                    percent_area=round((intersection_area/parking_area)*100,2)
                    text9="Percentage of parking area destroyed is: "+str(percent_area)+"%"+"\n"
                    output_text.insert("end",text9)
                    cv2.circle(image, strike_point, 10, (0, 0, 255), -1)        #strike point
                    cv2.circle(image, strike_point, parking_subdisp, (0, 255, 0), 1)
                    points,total_successful_submunition = generate_submunition_points(strike_point, parking_subdisp,parking_submun)
                    print("\n",total_successful_submunition,"out of ",parking_submun,"submunitions have successfully hit and ",parking_submun-total_successful_submunition," have failed ")
                    text5=str(total_successful_submunition)+" out of "+str(parking_submun)+" submunitions have successfully hit and "+str(parking_submun-total_successful_submunition)+" have failed "+"\n"
                    output_text.insert("end",text5)
                    sub_count_label.config(text="Submunition:"+str(total_successful_submunition))
                    for point in points:
                        cv2.circle(image, point, int(feet_to_pixels_width(parking_submun_damage_radius,scale_width)), (0, 255, 0), thickness=-1)
                    cv2.imshow('image',image)
                else:    
                    b=rect_point[1][1]
                    missile.append(1)
                    cv2.circle(image, (x, y), cep, (0, 0, 255), 1)
                    strike_point=generate_strike_point((x,y),.6*cep)
                    strike_point=np.round(strike_point).astype(int)
                    cv2.circle(image, strike_point, 10, (0, 0, 255), -1)        #strike point
                    cv2.circle(image, strike_point, subdisp, (0, 255, 0), 1)
                    points,total_successful_submunition = generate_submunition_points(strike_point, subdisp,submun)
                    print("\n",total_successful_submunition,"out of",submun,"submunitions have successfully hit and",submun-total_successful_submunition,"have failed ")
                    sub_count_label.config(text="Submunition:"+str(total_successful_submunition))
                    for point in points:
                        cv2.circle(image, point, int(feet_to_pixels_width(damage,scale_width)), (0, 255, 0), thickness=-1)
                    d1,u1=max_min_y(points)
                    if u1 < a and d1< a:
                        up_down_point.append((a,a))
                    elif u1 < a:
                        up_down_point.append((a,d1))
                    elif d1 > b and u1 >b:
                        up_down_point.append((b,b))
                    elif d1 > b:
                        up_down_point.append((u1,b))
                    else:
                        up_down_point.append((u1,d1))
                    diff_bw_max_min_y=diff(up_down_point)
                    up_down_point.pop()
                    count=0
                    for i in range(len(diff_bw_max_min_y)):
                        if diff_bw_max_min_y[i]>mos[1]:
                            count+=1
                    if count>0:
                        temp_message="\nNo of MOS strips remaining: "+str(count)
                    if count==0:
                        print("\nThe runway has been cut successfully!",len(missile),"missiles were required to cut the runway")
                        temp_message=temp_message+"\nThe runway has been cut successfully!"+str(len(missile))+" missiles were required to cut the runway"
                    if(count==0):
                        color="green"
                    else:
                        color="red"
                    message1=message+temp_message+"\n--------------------------------------------------------------------\n"
                    output_text.insert("end", message1)
                    output_text.see("end")
                    canvas1.create_oval(5, 5, 50, 50, fill= color)
                    missile_count_label.config(text="Missiles:"+str(len(missile)))
                    cv2.imshow('image',image)
            else:
                print("\nMissile has failed")
                missile.append(1)
                output_text.insert("end","\nMissile has failed\n--------------------------------------------------------------------\n")
                output_text.see("end")

def prepare_list_of_upper_down_points(cut_list_up_down_point_copy,a,b,x,y,cep): # (ONLY FOR GRAPH) To prepare list of up and down points of submunition
            strike_point = generate_strike_point((x, y), cep)
            strike_point = np.round(strike_point).astype(int)
            points,_ = generate_submunition_points(strike_point, subdisp, submun)
            d1, u1 = max_min_y(points)
            if u1 < a and d1< a:
                heapq.heappush(cut_list_up_down_point_copy, (a, a))
            elif u1 < a:
                heapq.heappush(cut_list_up_down_point_copy, (a, d1))
            elif d1 > b and u1 >b:
                heapq.heappush(cut_list_up_down_point_copy, (b, b))
            elif d1 > b:
                heapq.heappush(cut_list_up_down_point_copy, (u1, b))
            else:
                heapq.heappush(cut_list_up_down_point_copy, (u1, d1))

def prepare_diff_of_up_down_coordinate(cut_list_up_down_point): # (ONLY FOR GRAPH) To prepare difference of up and down points of submunition
    cut_list_up_down_point_sorted=[]
    diff_of_points=[]
    up_down_coordinate=heapq.heappop(cut_list_up_down_point)
    cut_list_up_down_point_sorted.append(up_down_coordinate)
    a1=up_down_coordinate[1]
    for i in range (len(cut_list_up_down_point)):
        up_down_coordinate1=heapq.heappop(cut_list_up_down_point)
        cut_list_up_down_point_sorted.append(up_down_coordinate1)
        diff=up_down_coordinate1[0]-a1
        a1=up_down_coordinate1[1]
        if diff < mos[1]:
            diff_of_points.append(0)
        else:
            diff_of_points.append(diff)
    return (diff_of_points,cut_list_up_down_point_sorted)



def cut_width(rect_top, rect_bottom,cross_cen , total_strike_point,cep):  # (ONLY FOR GRAPH) returns total no of missile required to cut particular CEP of runway along width
    cut_list_up_down_point = []
    diff_of_points = []
    a = rect_top[1]
    b = rect_bottom[1]
    cir_cor_new=[]
    cut_list_up_down_point_sorted1=[]
    no_of_missile=0
    cut_list_up_down_point.append((0, a))
    for i in range(total_strike_point):
        x, y = cross_cen[i]
        no_of_missile+=1
        prepare_list_of_upper_down_points(cut_list_up_down_point,a,b,x,y,cep)
    
    heapq.heappush(cut_list_up_down_point, (b, b))
    cut_list_up_down_point_copy=cut_list_up_down_point.copy()
    diff_of_points,_=prepare_diff_of_up_down_coordinate(cut_list_up_down_point)
    for i in range(len(diff_of_points)):
        if diff_of_points[i]:
            cir_cor_new.append(cross_cen[diff_of_points.index(diff_of_points[i])])

    while cir_cor_new:
        diff_of_points1=[]      
        for i in range(len(cir_cor_new)):
            x,y=cir_cor_new.pop()
            no_of_missile+=1
            prepare_list_of_upper_down_points(cut_list_up_down_point_copy,a,b,x,y,cep)
            
        cut_list_up_down_point_copy2=cut_list_up_down_point_copy.copy()       
        up_down_coordinate=heapq.heappop(cut_list_up_down_point_copy2)
        cut_list_up_down_point_sorted1.append(up_down_coordinate)
        a1=up_down_coordinate[1]
        for i in range (len(cut_list_up_down_point_copy2)):
            up_down_coordinate1=heapq.heappop(cut_list_up_down_point_copy2)
            cut_list_up_down_point_sorted1.append(up_down_coordinate1)
            diff=up_down_coordinate1[0]-a1
            a1=up_down_coordinate1[1]
            if diff < mos[1]:
                diff_of_points1.append(0)
            else:
                diff_of_points1.append(diff)
                cir_cor_new.append((x,y))
    return no_of_missile

def no_of_missile(cep): # (ONLY FOR GRAPH) To calculate no of missile required to cut runway along width
    total_no_of_missile=0
    total_no_of_missile+=cut_width(rect_point[0],rect_point[1],cross_cen,total_strike_point,cep)
    return total_no_of_missile

def graph(): # (ONLY FOR GRAPH) To plot graph
    colors = ['c', 'r', 'm','g','y']
    markers = ['D', 'x', 'o','^','s']
    missile_iteration=value4
    missile_each_graph=value5
    for cep in cep_values:
        result_x = []
        result_y = []
        cep1=int(cep*3.28 )
        prev=0
        for missile in range(1, int(missile_iteration)+1, 1):
            sum1 = 0
            for i in range(int(missile_each_graph)):
                no_missile = no_of_missile(cep1)
                if missile >= no_missile:
                    sum1 += 1
                if (sum1>prev) :
                    prev=sum1
            print('no of cut for missile', missile, '=', prev)
            result_x.append(missile)
            result_y.append(prev/(int(missile_each_graph))*100)
        plt.scatter(result_x, result_y, color=colors[cep_values.index(cep)], marker=markers[cep_values.index(cep)], s=30, label=f'CEP={cep1}')
        plt.plot(result_x, result_y, color=colors[cep_values.index(cep)])  # Line plot without label
    plt.xlabel("no of missile")
    plt.ylabel("Probability of single runway cut  %")
    plt.legend(loc='lower right')
    # plt.axis([0, int(value4), 0,100 ])    
    plt.xticks(np.arange(0,int(value4)+1,1))
    plt.show()

def graph_display(): # (ONLY FOR GRAPH) GUI component of graph
    def get_values():
        global value4
        global value5
        value1 = graph_entry1.get().strip()  # Remove leading/trailing spaces
        value2 = graph_entry2.get().strip()
        value3 = graph_entry3.get().strip()
        value4 = graph_entry4.get().strip()
        value5 = graph_entry5.get().strip()
        value6 = graph_entry6.get().strip()
        value7 = graph_entry7.get().strip()
        if value1 and value2 and value3 and value6 and value7:
            cep_values.append(int(value1))
            cep_values.append(int(value2))
            cep_values.append(int(value3))
            cep_values.append(int(value6))
            cep_values.append(int(value7))
            graph()
        else:
            print("Please enter values in all fields.")
            text8="Please enter values in all fields."
            output_text.insert("end",text8)

    graph_label=Label(graph_output,text="Give 5 CEP values (in meter)",font='Callibri 10 bold')
    graph_label.grid(row=0,column=0,padx=10,pady=10)

    graph_label1=Label(graph_output,text="cep 1 (meter)")
    graph_label1.grid(row=1,column=0,padx=10,pady=10)
    graph_entry1=Entry(graph_output)
    graph_entry1.insert(0,5)
    graph_entry1.grid(row=1,column=1)

    graph_label2=Label(graph_output,text="cep 2 (meter)")
    graph_label2.grid(row=3,column=0,padx=10,pady=10)
    graph_entry2=Entry(graph_output)
    graph_entry2.insert(0,25)
    graph_entry2.grid(row=3,column=1)
    
    graph_label3=Label(graph_output,text="cep 3 (meter)")
    graph_label3.grid(row=5,column=0,padx=10,pady=10)
    graph_entry3=Entry(graph_output)
    graph_entry3.insert(0,40)
    graph_entry3.grid(row=5,column=1)

    graph_label6=Label(graph_output,text="cep 4 (meter)")
    graph_label6.grid(row=7,column=0,padx=10,pady=10)
    graph_entry6=Entry(graph_output)
    graph_entry6.insert(0,200)
    graph_entry6.grid(row=7,column=1)

    graph_label7=Label(graph_output,text="cep 5 (meter)")
    graph_label7.grid(row=9,column=0,padx=10,pady=10)
    graph_entry7=Entry(graph_output)
    graph_entry7.insert(0,300)
    graph_entry7.grid(row=9,column=1)

    graph_label4=Label(graph_output,text="No of missiles")
    graph_label4.grid(row=11,column=0,padx=10,pady=10)
    graph_entry4=Entry(graph_output)
    graph_entry4.insert(0,10)
    graph_entry4.grid(row=11,column=1)

    graph_label5=Label(graph_output,text="No of iterations for each simulation")
    graph_label5.grid(row=13,column=0,padx=10,pady=10)
    graph_entry5=Entry(graph_output)
    graph_entry5.insert(0,100)
    graph_entry5.grid(row=13,column=1)

    button=Button(graph_output,text="Simulate Graph",command=get_values,font='Callibri 10 bold') 
    button.grid(row=15,column=0,padx=10,pady=5,sticky="news")


root = Tk()
root.title("Data entry Example")
root.geometry("900x700")
frame = ttk.Frame(root)
frame.grid(row=0, column=0, sticky="nsew")

# To implement scroll bar in GUI window
canvas = Canvas(frame)
scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
content_frame = ttk.Frame(canvas)
content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
def _on_mousewheel(event):
   canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

#missile info for runway
missile_frame1=LabelFrame(content_frame,text="Missile Information")
missile_frame1.grid(row=0,column=0,padx=20,pady=10,sticky="ew")

#missile info for parking area
missile_frame2=LabelFrame(content_frame,text="Missile Information [For Parking Area]")
missile_frame2.grid(row=3,column=0,padx=40,pady=10,sticky="ew")

# output window
missile_output=LabelFrame(content_frame,text="Output")
missile_output.grid(row=8,column=0,padx=20,pady=5,sticky="news")

#GUI Current status
visible_output=LabelFrame(content_frame,text="Current status")
visible_output.grid(row=7,column=0,padx=20,pady=4,sticky="news")

#graph frame
graph_output=LabelFrame(missile_output,text="graph")
graph_output.grid(row=1,column=1,padx=20,pady=5,sticky="news")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

canvas.create_window((0, 0), window=content_frame, anchor="nw")
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

#Labels for runway Area
cep_label=Label(missile_frame1,text="Enter CEP (in meters)")
cep_label.grid(row=0,column=0)
cep_entry=Entry(missile_frame1)
cep_entry.insert(0,30)
cep_entry.grid(row=0,column=1)

munition_pattern_label=Label(missile_frame1,text="Enter the submunition dispersal pattern radius(in feet)")
munition_pattern_label.grid(row=0,column=2)
munition_pattern_entry=Entry(missile_frame1)
munition_pattern_entry.insert(0,150)
munition_pattern_entry.grid(row=0,column=3)

no_munition_label=Label(missile_frame1,text="Enter the no of submunitions")
no_munition_label.grid(row=1,column=0)
no_munition_entry=Entry(missile_frame1)
no_munition_entry.insert(0,85)
no_munition_entry.grid(row=1,column=1)

cep_width_label=Label(missile_frame1,text="Enter the MOS width (in feet)")
cep_width_label.grid(row=1,column=2)
cep_width_entry=Entry(missile_frame1)
cep_width_entry.insert(0,100)
cep_width_entry.grid(row=1,column=3)

damage_label=Label(missile_frame1,text="damage radius (in feet)")
damage_label.grid(row=2,column=0)
damage_entry=Entry(missile_frame1)
damage_entry.insert(0,10)
damage_entry.grid(row=2,column=1)

runway_width_label=Label(missile_frame1,text="Enter the runway width (in feet)")
runway_width_label.grid(row=2,column=2)
runway_width_entry=Entry(missile_frame1)
runway_width_entry.insert(0,300)
runway_width_entry.grid(row=2,column=3)


#Labels for Parking Area
parking_cep_label=Label(missile_frame2,text="Enter CEP (in meters)")
parking_cep_label.grid(row=3,column=0)
parking_cep_entry=Entry(missile_frame2)
parking_cep_entry.insert(0,100)
parking_cep_entry.grid(row=3,column=1)

parking_munition_pattern_label=Label(missile_frame2,text="Enter the submunition dispersal pattern radius(in feet)")
parking_munition_pattern_label.grid(row=3,column=2)
parking_munition_pattern_entry=Entry(missile_frame2)
parking_munition_pattern_entry.insert(0,900)
parking_munition_pattern_entry.grid(row=3,column=3)

parking_no_munition_label=Label(missile_frame2,text="Enter the no of submunitions")
parking_no_munition_label.grid(row=4,column=0)
parking_no_munition_entry=Entry(missile_frame2)
parking_no_munition_entry.insert(0,820)
parking_no_munition_entry.grid(row=4,column=1)

parking_munition_radius_label=Label(missile_frame2,text="Enter the damage radius of each submunition(in feet)")
parking_munition_radius_label.grid(row=4,column=2)
parking_munition_radius_entry=Entry(missile_frame2)
parking_munition_radius_entry.insert(0,20)
parking_munition_radius_entry.grid(row=4,column=3)

parking_length_label=Label(missile_frame2,text="Enter the length of the parking area (in feet)")
parking_length_label.grid(row=5,column=0)
parking_length_entry=Entry(missile_frame2)
parking_length_entry.insert(0,2000)
parking_length_entry.grid(row=5,column=1)

parking_width_label=Label(missile_frame2,text="Enter the width of the parking area (in feet)")
parking_width_label.grid(row=5,column=2)
parking_width_entry=Entry(missile_frame2)
parking_width_entry.insert(0,1300)
parking_width_entry.grid(row=5,column=3)

for widget in missile_frame1.winfo_children():
    widget.grid_configure(padx=10,pady=10)

for widget in missile_frame2.winfo_children():
    widget.grid_configure(padx=10,pady=10)

output_text = Text(missile_output)
output_text.grid(row=1, column=0, sticky="nsew")

#cut status GUI
canvas1 = Canvas(visible_output, width=50, height=50)
canvas1.grid(row=0,column=0)
canvas1.create_oval(5, 5, 50, 50, fill= 'red')
cut_label=Label(visible_output,text="status")
cut_label.grid(row=1,column=0)
missile_image=Image.open("missile.png")
k_image=missile_image.resize((50,50),Image.BICUBIC)
tk_image=ImageTk.PhotoImage(k_image)
label=Label(visible_output,image=tk_image)
label.grid(row=0,column=3,padx=30)
missile_count_label=Label(visible_output,text="Missiles:"+str(0))
missile_count_label.grid(row=1,column=3)
submunition_image=Image.open("submunition.png")
resize_submunition=submunition_image.resize((50,50),Image.BICUBIC)
sub_image=ImageTk.PhotoImage(resize_submunition)
sub_label=Label(visible_output,image=sub_image)
sub_label.grid(row=0,column=4,padx=30)
sub_count_label=Label(visible_output,text="Submunitions:"+str(0))
sub_count_label.grid(row=1,column=4)

#displaying graph
button=Button(graph_output,text="display graph",command=graph_display,font='Callibri 10 bold') 
button.grid(row=0,column=0,padx=10,pady=5,sticky="news")

def main_function():
    global cep
    global subdisp
    global submun
    global mos
    global a
    global damage
    global parking_rectangle_points
    global parking_submun_damage_radius
    global parking_cep
    global parking_subdisp
    global parking_submun
    global total_strike_point
    global cross_cen
    # acquiring values from GUI
    cv2.namedWindow('image')
    cep = int(cep_entry.get())
    cep=int(feet_to_pixels_width(cep,scale_width))
    subdisp = int(munition_pattern_entry.get())
    subdisp=int(feet_to_pixels_width(subdisp,scale_width))
    submun = int(no_munition_entry.get())
    mos_width = int(cep_width_entry.get())
    damage=int(damage_entry.get())
    damage=feet_to_pixels_width(damage,scale_width)
    runway_width=int(runway_width_entry.get())
    parking_cep=int(parking_cep_entry.get())
    parking_cep=int(feet_to_pixels_width(parking_cep,parking_scale_length))
    parking_cep=int(parking_cep*3.28 )
    parking_subdisp=int(parking_munition_pattern_entry.get())
    parking_subdisp=feet_to_pixels_length(parking_subdisp,parking_scale_length)
    parking_submun=int(parking_no_munition_entry.get())
    parking_submun_damage_radius=int(parking_munition_radius_entry.get())
    parking_submun_damage_radius=feet_to_pixels_length(parking_submun_damage_radius,parking_scale_length)
    parking_area_length=int(parking_length_entry.get())
    parking_area_width=int(parking_width_entry.get())


    cep=int(cep*3.28 )
    rect_point=runway_draw(12000,runway_width)
    parking_rectangle_points=parking_rectangle(parking_area_length,parking_area_width)
    a = rect_point[0][1]
    up_down_point.append((0,a))
    mos=[5000,mos_width]  #MOS length 5000
    mos[0]=feet_to_pixels_length(mos[0],scale_length)
    mos[1]=feet_to_pixels_width(mos[1],scale_width)   
    _,_,total_strike_point,cross_cen=generate_strike_ideal_point_new(rect_point,mos,subdisp)
   
    cv2.setMouseCallback('image', on_mouse_click)

button=Button(content_frame,text="Enter data",command=main_function,font='Callibri 10 bold') 
button.grid(row=6,column=0,sticky="news",padx=20,pady=5)
root.mainloop()


cv2.waitKey(0)
cv2.destroyAllWindows()