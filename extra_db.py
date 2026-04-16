-- ================= INDEX =================
CREATE INDEX idx_student_name ON Students(name);
SHOW INDEX FROM Students;

-- ================= ALTER TABLE =================
ALTER TABLE Students ADD email VARCHAR(100);
ALTER TABLE Students MODIFY age INT NOT NULL;
ALTER TABLE Students DROP COLUMN email;

-- ================= UPDATE =================
UPDATE Results SET marks = 95 WHERE student_id = 1;

-- ================= DELETE =================
DELETE FROM Attendance WHERE percentage < 70;

-- ================= AGGREGATE FUNCTIONS =================
SELECT 
    COUNT(*) AS total_students,
    AVG(age) AS avg_age,
    MAX(age) AS oldest,
    MIN(age) AS youngest
FROM Students;

-- ================= GROUP BY =================
SELECT dept_id, COUNT(*) AS student_count
FROM Students
GROUP BY dept_id;

-- ================= ORDER BY + LIMIT =================
SELECT s.name, r.marks
FROM Results r
JOIN Students s ON r.student_id = s.student_id
ORDER BY r.marks DESC
LIMIT 3;

-- ================= DISTINCT =================
SELECT DISTINCT dept_id FROM Students;

-- ================= CONSTRAINT CHECK (WILL FAIL) =================
-- Uncomment to demonstrate constraint violation
-- INSERT INTO Students VALUES (10,'Test',10,1);

-- ================= TRANSACTION =================
START TRANSACTION;

UPDATE Results SET marks = 50 WHERE student_id = 1;

-- choose one:
ROLLBACK;  -- undo changes
-- COMMIT; -- save changes

-- ================= STORED PROCEDURE =================
DELIMITER //

CREATE PROCEDURE GetStudentMarks(IN sid INT)
BEGIN
    SELECT s.name, r.marks
    FROM Students s
    JOIN Results r ON s.student_id = r.student_id
    WHERE s.student_id = sid;
END //

DELIMITER ;

-- call procedure
CALL GetStudentMarks(1);

-- ================= TRIGGER =================
DELIMITER //

CREATE TRIGGER check_marks
BEFORE INSERT ON Results
FOR EACH ROW
BEGIN
    IF NEW.marks < 0 OR NEW.marks > 100 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Invalid marks';
    END IF;
END //

DELIMITER ;
