-- CREATE DATABASE
CREATE DATABASE student_db;
USE student_db;

-- TABLES
CREATE TABLE Departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

CREATE TABLE Courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

CREATE TABLE Instructors (
    instructor_id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

CREATE TABLE Enrollments (
    enroll_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE Grades (
    grade_id INT PRIMARY KEY,
    enroll_id INT,
    grade CHAR(2),
    FOREIGN KEY (enroll_id) REFERENCES Enrollments(enroll_id)
);

CREATE TABLE Attendance (
    attendance_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    percentage DECIMAL(5,2)
);

CREATE TABLE Timetable (
    timetable_id INT PRIMARY KEY,
    course_id INT,
    instructor_id INT,
    time_slot VARCHAR(50)
);

CREATE TABLE Fees (
    fee_id INT PRIMARY KEY,
    student_id INT,
    amount DECIMAL(10,2)
);

CREATE TABLE Library (
    book_id INT PRIMARY KEY,
    book_name VARCHAR(50)
);

CREATE TABLE Library_Issues (
    issue_id INT PRIMARY KEY,
    student_id INT,
    book_id INT
);

-- INSERT DATA (5 rows each)
INSERT INTO Departments VALUES
(1,'CSE'),(2,'ECE'),(3,'EEE'),(4,'MECH'),(5,'CIVIL');

INSERT INTO Students VALUES
(1,'Asha',1),(2,'Bala',2),(3,'Charan',1),(4,'Divya',3),(5,'Eshan',4);

INSERT INTO Courses VALUES
(101,'DBMS',1),(102,'Networks',1),(103,'Circuits',2),(104,'Thermo',4),(105,'Structures',5);

INSERT INTO Instructors VALUES
(1,'Dr Rao',1),(2,'Dr Kumar',2),(3,'Dr Singh',3),(4,'Dr Mehta',4),(5,'Dr Iyer',5);

INSERT INTO Enrollments VALUES
(1,1,101),(2,2,103),(3,3,102),(4,4,103),(5,5,104);

INSERT INTO Grades VALUES
(1,1,'A'),(2,2,'B'),(3,3,'A'),(4,4,'C'),(5,5,'B');

INSERT INTO Attendance VALUES
(1,1,101,90),(2,2,103,85),(3,3,102,88),(4,4,103,70),(5,5,104,75);

INSERT INTO Timetable VALUES
(1,101,1,'9AM'),(2,102,1,'10AM'),(3,103,2,'11AM'),(4,104,4,'12PM'),(5,105,5,'1PM');

INSERT INTO Fees VALUES
(1,1,50000),(2,2,48000),(3,3,50000),(4,4,47000),(5,5,46000);

INSERT INTO Library VALUES
(1,'DBMS Book'),(2,'Networks Book'),(3,'Circuits Book'),(4,'Thermo Book'),(5,'Civil Book');

INSERT INTO Library_Issues VALUES
(1,1,1),(2,2,3),(3,3,2),(4,4,3),(5,5,4);

-- JOINS
-- INNER JOIN
SELECT s.name, c.course_name
FROM Students s
INNER JOIN Enrollments e ON s.student_id = e.student_id
INNER JOIN Courses c ON e.course_id = c.course_id;

-- LEFT JOIN
SELECT s.name, f.amount
FROM Students s
LEFT JOIN Fees f ON s.student_id = f.student_id;

-- RIGHT JOIN
SELECT c.course_name, t.time_slot
FROM Timetable t
RIGHT JOIN Courses c ON t.course_id = c.course_id;

-- FULL JOIN (MySQL workaround using UNION)
SELECT s.name, f.amount
FROM Students s
LEFT JOIN Fees f ON s.student_id = f.student_id
UNION
SELECT s.name, f.amount
FROM Students s
RIGHT JOIN Fees f ON s.student_id = f.student_id;

-- CROSS JOIN
SELECT s.name, d.dept_name
FROM Students s
CROSS JOIN Departments d;

-- SUBQUERIES
-- 1
SELECT name FROM Students
WHERE student_id IN (
    SELECT student_id FROM Enrollments WHERE course_id = 101
);

-- 2
SELECT name FROM Students
WHERE student_id = (
    SELECT student_id FROM Fees ORDER BY amount DESC LIMIT 1
);

-- 3
SELECT course_name FROM Courses
WHERE course_id IN (
    SELECT course_id FROM Enrollments GROUP BY course_id HAVING COUNT(*) > 1
);

-- VIEW
CREATE VIEW Student_Details AS
SELECT s.name, d.dept_name, f.amount
FROM Students s
JOIN Departments d ON s.dept_id = d.dept_id
JOIN Fees f ON s.student_id = f.student_id;
