# Unit Tests

Importing functions from within this folder, as in
`from adlframework.filters.general_filters import min_array_shape`,
seems to cause errors and bugs that result in incorrect outputs.
For example, the tests pass in testing_general_filters.py, but not in
unit_tests/TestGeneralFilters.py.

For now, use the temporary testing_x.py files found outside this unit_tests folder.
