drop database if exists turmeric_leaf;
create database turmeric_leaf;
use turmeric_leaf;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    email VARCHAR(50), 
    password VARCHAR(50)
    );
