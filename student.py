1. ER Diagram (Concept)

STUDENT ---< ENROLLMENT >--- COURSE
   |                            |
   |                            |
DEPARTMENT                INSTRUCTOR

2. CREATE TABLES

CREATE TABLE Department (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE Student (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Department(dept_id)
);

CREATE TABLE Instructor (
    instructor_id INT PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE Course (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(50),
    instructor_id INT,
    FOREIGN KEY (instructor_id) REFERENCES Instructor(instructor_id)
);

CREATE TABLE Enrollment (
    enroll_id INT PRIMARY KEY,
    student_id INT,
    course_id INT,
    grade CHAR(2),
    FOREIGN KEY (student_id) REFERENCES Student(student_id),
    FOREIGN KEY (course_id) REFERENCES Course(course_id)
);

3. INSERT DATA

INSERT INTO Department VALUES (1, 'CSE'), (2, 'ECE');

INSERT INTO Student VALUES
(101, 'Anu', 1),
(102, 'Bala', 1),
(103, 'Charan', 2),
(104, 'Divya', 2),
(105, 'Esha', 1);

INSERT INTO Instructor VALUES
(201, 'Dr. Rao'),
(202, 'Dr. Mehta');

INSERT INTO Course VALUES
(301, 'DBMS', 201),
(302, 'Networks', 202),
(303, 'AI', 201);

INSERT INTO Enrollment VALUES
(1, 101, 301, 'A'),
(2, 102, 301, 'B'),
(3, 103, 302, 'A'),
(4, 104, 303, 'B'),
(5, 105, 301, 'A');

4. JOINS

INNER JOIN:
SELECT s.name, c.course_name
FROM Student s
INNER JOIN Enrollment e ON s.student_id = e.student_id
INNER JOIN Course c ON e.course_id = c.course_id;

LEFT JOIN:
SELECT s.name, e.course_id
FROM Student s
LEFT JOIN Enrollment e ON s.student_id = e.student_id;

RIGHT JOIN:
SELECT s.name, e.course_id
FROM Student s
RIGHT JOIN Enrollment e ON s.student_id = e.student_id;

FULL JOIN (Simulated):
SELECT s.name, e.course_id
FROM Student s
LEFT JOIN Enrollment e ON s.student_id = e.student_id
UNION
SELECT s.name, e.course_id
FROM Student s
RIGHT JOIN Enrollment e ON s.student_id = e.student_id;

5. SUBQUERIES

Students with A grade:
SELECT name FROM Student
WHERE student_id IN (
    SELECT student_id FROM Enrollment WHERE grade = 'A'
);

Courses by Dr. Rao:
SELECT course_name FROM Course
WHERE instructor_id = (
    SELECT instructor_id FROM Instructor WHERE name = 'Dr. Rao'
);

6. VIEW

CREATE VIEW Student_Course_View AS
SELECT s.name, c.course_name, e.grade
FROM Enrollment e
JOIN Student s ON e.student_id = s.student_id
JOIN Course c ON e.course_id = c.course_id;

SELECT * FROM Student_Course_View;
