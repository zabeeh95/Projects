clear all
clf

delete(instrfindall);
w = serial('COM4');
set(w,'DataBits',8);
set(w,'StopBits',1);
set(w,'BaudRate',9600);
set(w,'Timeout',10);
set(w,'Parity','none');
fopen(w);
fprintf(' ');

tic

%PLOTTING & ANIMATION
xlim auto
%array 01
subplot(3,1,1);
grid on;
h1 = animatedline; 
subplot(3,1,2);
grid on;
h2 = animatedline('Color','blue');
subplot(3,1,3);
grid on;
h3 = animatedline('Color','green');

i=1; 

serial_data = '' ;
data = [];
valueA1(i,1) =550; valueA2(i,2)=550; valueA3(i,3)=550;

while true
    
    serial_data = fgets(w);
    data = strsplit(serial_data);
    numeric_data=str2double(data);
    if length(numeric_data) ~= 5
        continue
    end
    
    for k = [2 3 4]
        if isnan(numeric_data(k))
            numeric_data(k) = 550;
        end
    end
        for g = [2 3 4]
            if numeric_data(g)>1000 || numeric_data(g)< 300 %ambient noise calibration
                numeric_data(g) = 550;
            end
        end
        
       if numeric_data(1) == 1
            
    sound_data_ZB_1(1,i) = numeric_data(2);
    timeA1(i,1)=toc;
    valueA1(i,1)=sound_data_ZB_1(1,i);
    
   
    sound_data_ZB_1(2,i) = numeric_data(3);
    timeA2(i,1)=toc;
    valueA2(i,2)=sound_data_ZB_1(2,i);
       
    
    sound_data_ZB_1(3,i) = numeric_data(4);
    timeA3(i,1)=toc;
    valueA3(i,3)=sound_data_ZB_1(3,i);
    
    
    addpoints(h1,timeA1(i,1),valueA1(i,1));
    addpoints(h2,timeA2(i,1),valueA2(i,2));
    addpoints(h3,timeA3(i,1),valueA3(i,3));
        

 if timeA1(i,1)-180 < 0
     
        xlim([0 timeA1(i,1)]);
        subplot(3,1,1);
        xlim([0 timeA2(i,1)]);
        subplot(3,1,2); 
        xlim([0 timeA3(i,1)]);
        subplot(3,1,3);
    else
        xlim([timeA1(i,1)-180 timeA3(i,1)]);
        subplot(3,1,1);
        xlim([timeA2(i,1)-180 timeA1(i,1)]);
        subplot(3,1,2); 
        xlim([timeA3(i,1)-180 timeA2(i,1)]);
        subplot(3,1,3); 
 end
 
 i = i+1 ;
  
       
       end         
      
drawnow
end

fclose(w)
    
    clearpoints(h1)
    clearpoints(h2)
    clearpoints(h3)

    