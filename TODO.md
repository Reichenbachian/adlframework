Change output options to allow non-verbose

Add timeout for data fetching

Make or check if data loading is entirely generator, not loaded all at once

Add in visualization library:
	https://github.com/raghakot/keras-vis

Create an experiments database with the following format.
	### Experiment Database
	The experiment database is organized in the following way.

	Project Table
	id - a unique project id.
	name - a project name. Should really be unique
	Users - A list of users working on project.

	## Experiments Tables
	### Experiment Table
		id - a unique user id
		user_name - preferred user name
		email - preferred user email
		epoch - Final reached epoch
		number of entities - number of data entitites used
		data_source - hashes of datasources
		processors - a list of the processors used hashes
		filters - a list of the filters used hashes
		augmentors - a list of the augmentors used hashes
		Callbacks - 

	## Results
	### Callbacks

	## Preprocessing Tables
	### Names of Datasources
		id - a unique datasource id


	## Preprocessing Tables
	### Filter Table
		id - a unique filter id
		code - actual python code
		hash - hash of filter's code.
		date created - creation date

	### Augmentors Table
		id - a unique filter id
		code - actual python code
		hash - hash of filter's code.
		date created - creation date

	### Processors Table
		id - a unique filter id
		code - actual python code
		hash - hash of filter's code.
		date created - creation date

	### Callback Table
		id - a unique filter id
		code - actual python code
		hash - hash of filter's code.
		date created - creation date

	### Nets Table
		id - a unique filter id
		code - actual python code
		hash - hash of filter's code.
		date created - creation date

	## Administration Tables
	### Users Table
		id - a unique user id
		name - preferred user name
		email - preferred user email


	### Network
	s3 wrapper