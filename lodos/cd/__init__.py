"""
Contact dynamics: the constraint solve over Lodos's and MJWarp's contacts.

`solver` is MJWarp's constraint solver, copied in; `driver` hands it Lodos's contacts and drives it; `groups` gathers
a pair's contacts into contact groups, `reduce` keeps a few contacts of each pair,
`rowspace` and `fused` solve the Newton system the rows' way or in one kernel per world,
and `params` and `bins` hold the pairs' parameters and the direction bins.
"""
