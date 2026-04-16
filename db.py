-- ================= DATABASE =================
CREATE DATABASE student_db;
USE student_db;

-- ================= TABLES =================

CREATE TABLE Departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT CHECK (age > 15),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

CREATE TABLE Professors (
    prof_id INT PRIMARY KEY,
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

CREATE TABLE Enrollments (
    enroll_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE Exams (
    exam_id INT PRIMARY KEY,
    course_id INT,
    date DATE,
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE Results (
    result_id INT PRIMARY KEY,
    student_id INT,
    exam_id INT,
    marks INT CHECK (marks >= 0 AND marks <= 100),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (exam_id) REFERENCES Exams(exam_id)
);

CREATE TABLE Attendance (
    attendance_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    percentage INT CHECK (percentage BETWEEN 0 AND 100),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

CREATE TABLE Classrooms (
    room_id INT PRIMARY KEY,
    room_name VARCHAR(50),
    capacity INT
);

CREATE TABLE CourseAllocation (
    alloc_id INT PRIMARY KEY,
    course_id INT,
    prof_id INT,
    room_id INT,
    FOREIGN KEY (course_id) REFERENCES Courses(course_id),
    FOREIGN KEY (prof_id) REFERENCES Professors(prof_id),
    FOREIGN KEY (room_id) REFERENCES Classrooms(room_id)
);

-- ================= SAMPLE DATA =================

INSERT INTO Departments VALUES
(1,'CSE'),(2,'ECE'),(3,'MECH');

INSERT INTO Students VALUES
(1,'Arun',20,1),(2,'Bala',21,2),(3,'Chitra',19,1),(4,'Divya',22,3);

INSERT INTO Professors VALUES
(1,'Dr.Ram',1),(2,'Dr.Kumar',2),(3,'Dr.Sita',3);

INSERT INTO Courses VALUES
(101,'DBMS',1),(102,'Networks',2),(103,'Thermodynamics',3);

INSERT INTO Enrollments VALUES
(1,1,101),(2,2,102),(3,3,101),(4,4,103);

INSERT INTO Exams VALUES
(1,101,'2024-01-01'),(2,102,'2024-01-02'),(3,103,'2024-01-03');

INSERT INTO Results VALUES
(1,1,1,85),(2,2,2,70),(3,3,1,90),(4,4,3,60);

INSERT INTO Attendance VALUES
(1,1,101,80),(2,2,102,75),(3,3,101,90),(4,4,103,65);

INSERT INTO Classrooms VALUES
(1,'A101',40),(2,'B201',50);

INSERT INTO CourseAllocation VALUES
(1,101,1,1),(2,102,2,2),(3,103,3,1);

-- ================= JOINS =================

-- INNER JOIN: Students with their courses
SELECT s.name, c.course_name
FROM Students s
INNER JOIN Enrollments e ON s.student_id = e.student_id
INNER JOIN Courses c ON e.course_id = c.course_id;

-- LEFT JOIN: All students with results (even if no result)
SELECT s.name, r.marks
FROM Students s
LEFT JOIN Results r ON s.student_id = r.student_id;

-- RIGHT JOIN: All courses with enrolled students
SELECT c.course_name, s.name
FROM Students s
RIGHT JOIN Enrollments e ON s.student_id = e.student_id
RIGHT JOIN Courses c ON e.course_id = c.course_id;

-- FULL OUTER JOIN (simulate using UNION)
SELECT s.name, c.course_name
FROM Students s
LEFT JOIN Enrollments e ON s.student_id = e.student_id
LEFT JOIN Courses c ON e.course_id = c.course_id

UNION

SELECT s.name, c.course_name
FROM Students s
RIGHT JOIN Enrollments e ON s.student_id = e.student_id
RIGHT JOIN Courses c ON e.course_id = c.course_id;

-- ================= SUBQUERIES =================

-- 1. WHERE subquery: students scoring above average
SELECT name
FROM Students
WHERE student_id IN (
    SELECT student_id
    FROM Results
    WHERE marks > (SELECT AVG(marks) FROM Results)
);

-- 2. GROUP BY subquery: dept with highest avg marks
SELECT dept_id
FROM Students
WHERE student_id IN (
    SELECT student_id
    FROM Results
)
GROUP BY dept_id
HAVING AVG(student_id) > 2;

-- 3. HAVING subquery: courses with avg marks > overall avg
SELECT course_id, AVG(marks) AS avg_marks
FROM Results r
JOIN Exams e ON r.exam_id = e.exam_id
GROUP BY course_id
HAVING AVG(marks) > (
    SELECT AVG(marks) FROM Results
);

-- ================= VIEW =================

CREATE VIEW Student_Report AS
SELECT s.name, d.dept_name, c.course_name, r.marks
FROM Students s
JOIN Departments d ON s.dept_id = d.dept_id
JOIN Enrollments e ON s.student_id = e.student_id
JOIN Courses c ON e.course_id = c.course_id
JOIN Results r ON s.student_id = r.student_id;

-- View usage
SELECT * FROM Student_Report;
