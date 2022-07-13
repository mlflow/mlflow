CREATE DATABASE mlflowdb;
GO

USE mlflowdb;

CREATE LOGIN mlflowuser
    WITH PASSWORD = 'Mlfl*wpassword1';
GO

CREATE USER mlflowuser FOR LOGIN mlflowuser;
GO

ALTER ROLE db_owner
    ADD MEMBER mlflowuser;
GO
