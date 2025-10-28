SELECT COUNT(*) AS n_customers FROM customers;
SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC;
SELECT id, name FROM customers WHERE country = 'Italy' ORDER BY id LIMIT 5;
