  PROGRAM plot_winds
    !
    ! --- REGRID/MM5 gives us sea surface winds 
    ! --- They are in ASCII format
    ! --- This code will employ GrADS to plot them
    ! ---
    ! --- Dr. Gao Shanhong, 09 April 2007.
    ! ---
    !
    implicit none
    !
    character(len=13)  :: date_time
    character(len=50)  :: speed
    integer            :: kx, ky, len_file
    real, allocatable, dimension(:,:) :: u0, v0, u1, v1, u2, v2, u3, v3
    real               :: res, lon_beg, lat_beg, max_spd
    !
    integer                           :: nargum, iargc, k
    character(len=100), dimension(10) :: argum
    !
    character(len=30)  :: temp_cha
    real               :: x1, x2, y1, y2
    !
    ! === Step 1: get input file
    nargum=iargc()
    do k = 1, nargum
      call getarg(k,argum(k))
    enddo
    !write(*,*) ''
    !write(*,*) ' ======================================'
    !write(*,*) '  To plot sea surface winds from'
    !write(*,*) '  REGRID/MM5 in ASCII long-lati Grid! '
    !write(*,*) ''
    !write(*,*) '  Dr. GAO Shanhong, 09 April 2005.'
    !write(*,*) ' ======================================'
    !write(*,*) ''
    if( nargum == 0 ) then
      write(*,*) ' Usage:  '
      write(*,*) '   plot_winds.exe YYYYMMDD_00.sfc YYYYMMDD_03.sfc'
      write(*,*) ' or  '
      write(*,*) '   plot_winds.exe YYYYMMDDHHSW.dat'
      write(*,*) ' '
      stop
    endif 
    !
    ! === Step 2: read input file
    open(1, file=trim(argum(1)), status='old', action='read' )
      read(1,*) temp_cha, temp_cha, date_time
      read(1,*) temp_cha, temp_cha, kx, temp_cha, ky 
      read(1,*) temp_cha, temp_cha, temp_cha, lon_beg
      read(1,*) temp_cha, temp_cha, temp_cha, lat_beg
      read(1,*) temp_cha, temp_cha, res
      read(1,*)
      read(1,*)
      read(1,'(a50)') speed
      read(1,*)
      !
      allocate( u0(kx,ky), v0(kx,ky) )
      allocate( u1(kx,ky), v1(kx,ky) )
      allocate( u2(kx,ky), v2(kx,ky) )
      allocate( u3(kx,ky), v3(kx,ky) )
      read(1,*)
      read(1,'(15f7.2)') u0(:,:)
      read(1,*)
      read(1,'(15f7.2)') v0(:,:)
    close(1)

    open(1, file=trim(argum(2)), status='old', action='read' )
      read(1,*) temp_cha, temp_cha, date_time
      read(1,*) temp_cha, temp_cha, kx, temp_cha, ky 
      read(1,*) temp_cha, temp_cha, temp_cha, lon_beg
      read(1,*) temp_cha, temp_cha, temp_cha, lat_beg
      read(1,*) temp_cha, temp_cha, res
      read(1,*)
      read(1,*)
      read(1,'(a50)') speed
      read(1,*)
      !
      read(1,*)
      read(1,'(15f7.2)') u3(:,:)
      read(1,*)
      read(1,'(15f7.2)') v3(:,:)
    close(1)
    ! 
    u1 = u0 * (2/3.0) + u3 * (1/3.0)
    v1 = v0 * (2/3.0) + v3 * (1/3.0)

    u2 = u0 * (1/3.0) + u3 * (2/3.0)
    v2 = v0 * (1/3.0) + v3 * (2/3.0)

    print*,u1(1,1),u0(1,1),u3(1,1)
    print*,u1(2,2),u0(2,2),u3(2,2)
   ! === Step 3: create data for GrADS plotting
   ! --- write out
   max_spd = SQRT( maxval( u1*u1 + v1*v1 ) )
   open(1, file='sfc_wind_ok1', action='write' )
     write(1,'(a)') ' DATE =  YYYY-MM-DD:HH   UTC'
     write(1,'(a,i4,a,i4,a)') ' Dimension = ', kx, ' X ', ky, ' ( West-east X South-North Direction )'
     write(1,'(a,f8.2,a)') ' Longitude domain = ', lon_beg, 'lon_end,  degree'
     write(1,'(a,f8.2,a)') ' Latitude  domain = ', lat_beg, 'lat_end,  degree'
     write(1,'(a,f7.2,a)')  ' Resolution       = ', res, ' degree'
     write(1,'(a)') " FORMAT = ASCII, 15 number every line"
     write(1,'(a)') " FORMAT of Fortran90 writing: write(9,'(15f7.2)')  array(:,:)"
     write(1,'(a,F7.2,a)') ' MAX-WINDS: ', max_spd  ,' ; U10 and V10 Unit: m/s'
     write(1,*) ''
     !
     write(1,'(a)') ' FIELD = U10'
     write(1,'(15f7.2)') u1(:,:)
     write(1,'(a)') ' FIELD = V10'
     write(1,'(15f7.2)') v1(:,:)
   close(1)
   ! ====
   ! --- write out
   max_spd = SQRT( maxval( u2*u2 + v2*v2 ) )
   open(1, file='sfc_wind_ok2', action='write' )
     write(1,'(a)') ' DATE =  YYYY-MM-DD:HH   UTC'
     write(1,'(a,i4,a,i4,a)') ' Dimension = ', kx, ' X ', ky, ' ( West-east X South-North Direction )'
     write(1,'(a,f8.2,a)') ' Longitude domain = ', lon_beg, 'lon_end,  degree'
     write(1,'(a,f8.2,a)') ' Latitude  domain = ', lat_beg, 'lat_end,  degree'
     write(1,'(a,f7.2,a)')  ' Resolution       = ', res, ' degree'
     write(1,'(a)') " FORMAT = ASCII, 15 number every line"
     write(1,'(a)') " FORMAT of Fortran90 writing: write(9,'(15f7.2)')  array(:,:)"
     write(1,'(a,F7.2,a)') ' MAX-WINDS: ', max_spd  ,' ; U10 and V10 Unit: m/s'
     write(1,*) ''
     !
     write(1,'(a)') ' FIELD = U10'
     write(1,'(15f7.2)') u2(:,:)
     write(1,'(a)') ' FIELD = V10'
     write(1,'(15f7.2)') v2(:,:)
   close(1)

    !
    !
  END PROGRAM plot_winds
