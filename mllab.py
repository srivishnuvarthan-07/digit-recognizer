_________________________________________________________________
1st conditional statements
_________________________________________________________________

public class GradeChecker {
    public static void main(String[] args) {
        int marks = 75; 
        if (marks >= 85) {
            System.out.println("Grade: A");
        } else if (marks >= 60) {
            System.out.println("Grade: B");
        } else {
            System.out.println("Grade: Fail");
        }
    }
}

_________________________________________________________________
2nd Method Overloading & Constructor Overloading
_________________________________________________________________

public class AreaCalculator {

    // Constructor Overloading
    AreaCalculator() {
        System.out.println("Default constructor called.");
    }

    AreaCalculator(String shape) {
        System.out.println("Calculating area for: " + shape);
    }

    // Method Overloading

    double area(double side) {
        return side * side;
    }

    double area(double length, double width) {
        return length * width;
    }

    double area(float radius) {
        return 3.14159f * radius * radius;
    }

    public static void main(String[] args) {
        // Constructor overloading
        AreaCalculator obj1 = new AreaCalculator();
        AreaCalculator obj2 = new AreaCalculator("Circle");

        System.out.println("Area of square: " + obj1.area(5));
        System.out.println("Area of rectangle: " + obj1.area(5, 3));
        System.out.println("Area of circle: " + obj1.area(4.5f));
    }
}

__________________________________________________________________
3rd Employee Details using scanner class
__________________________________________________________________

import java.util.Scanner;

public class EmployeeDetails {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter Employee ID: ");
        int id = sc.nextInt();

        sc.nextLine(); // consume newline

        System.out.print("Enter Employee Name: ");
        String name = sc.nextLine();

        System.out.print("Enter Salary: ");
        double salary = sc.nextDouble();

        System.out.println("\n--- Employee Details ---");
        System.out.println("ID: " + id);
        System.out.println("Name: " + name);
        System.out.println("Salary: ₹" + salary);
        
        sc.close();
    }
}

_________________________________________________________________
4th String operations
_________________________________________________________________

public class StringOperations {
    public static void main(String[] args) {
        String str1 = "Hello";
        String str2 = "World";

        // Concatenation
        String result = str1 + " " + str2;
        System.out.println("Concatenated String: " + result);

        // Substring
        String sub = result.substring(0, 5);
        System.out.println("Substring (0 to 4): " + sub);

        // Length
        System.out.println("Length of result: " + result.length());

        // Comparison
        String str3 = "hello";
        System.out.println("str1 equals str3? " + str1.equals(str3));
        System.out.println("str1 equalsIgnoreCase str3? " + str1.equalsIgnoreCase(str3));
    }
}

______________________________________________________________________________
5th Abstract Class
______________________________________________________________________________

abstract class Shape {
    abstract double area(); // abstract method
}
class Circle extends Shape {
    double radius;
    Circle(double r) {
        radius = r;
    }
    double area() {
        return Math.PI * radius * radius;
    }
}
class Rectangle extends Shape {
    double length, width;
    Rectangle(double l, double w) {
        length = l;
        width = w;
    }
    double area() {
        return length * width;
    }
}
public class Main {
    public static void main(String[] args) {
        Shape c = new Circle(3);
        Shape r = new Rectangle(4, 5);

        System.out.println("Circle Area: " + c.area());
        System.out.println("Rectangle Area: " + r.area());
    }
}

___________________________________________________________________________
6th Inheritance
___________________________________________________________________________

class Person {
    String name;
    int age;
    void display() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
    }
}
class Student extends Person {
    int rollNo;
    void show() {
        display(); // call base class method
        System.out.println("Roll No: " + rollNo);
    }
}
public class Main {
    public static void main(String[] args) {
        Student s = new Student();
        s.name = "Alice";
        s.age = 20;
        s.rollNo = 101;
        s.show();
    }
}

___________________________________________________________________
7th Interfaces
___________________________________________________________________

interface Vehicle {
    void start(); // abstract method
}
class Car implements Vehicle {
    public void start() {
        System.out.println("Car started");
    }
}
class Bike implements Vehicle {
    public void start() {
        System.out.println("Bike started");
    }
}
public class Main {
    public static void main(String[] args) {
        Vehicle v1 = new Car();
        Vehicle v2 = new Bike();

        v1.start();  
        v2.start();  
    }
}

_____________________________________________________________________
8th Exception handling and threads
_____________________________________________________________________

class MyThread extends Thread {
    public void run() {
        System.out.println("Thread is running...");
    }
}
public class Main {
    public static void main(String[] args) {
        // Exception Handling
        try {
            int result = 10 / 0;  // Will cause ArithmeticException
        } catch (ArithmeticException e) {
            System.out.println("Caught Exception: " + e);
        }

        // Multithreading
        MyThread t = new MyThread();
        t.start();  // Start the thread
    }
}


_____________________________________________________________________
9th File operations
_____________________________________________________________________

import java.io.*;
public class FileDemo {
    public static void main(String[] args) {
        try {           
            File file = new File("demo.txt");
            file.createNewFile();
           
            FileWriter writer = new FileWriter(file);
            writer.write("Hello, File Handling!");
            writer.close();
            
            FileReader reader = new FileReader(file);
            int ch;
            while ((ch = reader.read()) != -1) {
                System.out.print((char) ch);
            }
            reader.close();
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
}

__________________________________________________________
10th JDBC with MySql
__________________________________________________________

import java.sql.*;
public class FileDemo {
    public static void main(String[] args) {
        // Database connection details
        String url = "jdbc:mysql://localhost:3306/todo_app"; // replace 'yourDatabase' with your DB name
        String username = "root"; // replace with your MySQL username
        String password = "vishnu"; // replace with your MySQL password

        // Establish connection
        try (Connection conn = DriverManager.getConnection(url, username, password)) {
            System.out.println("Connected to the database!");

            // Create a Statement
            Statement stmt = conn.createStatement();

            // Execute a query to retrieve data
            String query = "SELECT * FROM todo"; // replace 'yourTable' with your table name
            ResultSet rs = stmt.executeQuery(query);

            // Process the result
            while (rs.next()) {
                System.out.println("ID: " + rs.getInt("id") + ", Name: " + rs.getString("task"));
            }
        } catch (SQLException e) {
            System.out.println("Database connection failed: " + e.getMessage());
        }
    }
}

//Data base Query
CREATE DATABASE yourDatabase;
USE yourDatabase;

CREATE TABLE yourTable (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

INSERT INTO yourTable (id, name) VALUES (1, 'John Doe');
INSERT INTO yourTable (id, name) VALUES (2, 'Jane Doe');





-------------------------------------------------------------------END-----------------------------------------------------------------------




