 PROGRAM read_sfc_uv
   !
   implicit none
   !
   ! --- let this code allocatable !
   ! --- 16 Feb 2011.
   ! --------------------------------------------
   !
   integer :: kx, ky
   real    :: lon_beg, lon_end, lat_beg, lat_end, res, max_spd
   real, allocatable, dimension(:,:)  :: u, v
   integer :: k, j

   ! --- get dimension 
   open(1, file='dimension.dat', status='old')
     read(1,*) kx, ky, lon_beg, lon_end, lat_beg, lat_end, res
   close(1)

   ! --- read sfc_wind
   allocate( u(kx,ky), v(kx,ky) )
   open(1,file='sfc_wind')
     read(1,*) k, j
     read(1,*) u
     read(1,*) k, j
     read(1,*) v
   close(1)
   !u = u * 1.15
   !v = v * 1.15
   max_spd = SQRT( maxval( u*u + v*v ) )

   ! --- write out
   open(1, file='sfc_wind_ok', action='write' )
     write(1,'(2f8.2)')  lon_beg, lon_end
     write(1,'(2f8.2)')  lat_beg, lat_end
     write(1,'(f7.2)')   res
     write(1,'(a)') " FORMAT of Fortran90 writing: write(9,'(15f7.2)')  array(:,:)"
     !
     write(1,'(a)') ' FIELD = U10'
     write(1,'(15f7.2)') u(:,:)
     write(1,'(a)') ' FIELD = V10'
     write(1,'(15f7.2)') v(:,:)
   close(1)


 END PROGRAM read_sfc_uv
