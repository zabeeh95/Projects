
int mic01 = A0;
int mic02 = A2;
int mic03 = A4;// select the input pin for the potentiometer

int mic01Value = 0.0;  // variable to store the value coming from the sensor
int mic02Value = 0.0;
int mic03Value = 0.0;

void setup() {
  Serial.begin(9600);
  
}

void loop() 
{ 
                                      
  mic01Value = analogRead(mic01);
  mic02Value = analogRead(mic02);
  mic03Value = analogRead(mic03);
 

 Serial.print("1 "); Serial.print(mic01Value);Serial.print(" ");Serial.print(mic02Value);  Serial.print(" "); Serial.println(mic03Value); 

  
}

