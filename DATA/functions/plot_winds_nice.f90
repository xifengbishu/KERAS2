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
    character(len=9)   :: max_spd
    integer            :: kx, ky, len_file
    real, allocatable, dimension(:,:) :: u, v
    real               :: res, lon_beg, lat_beg
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
      write(*,*) '   plot_winds.exe YYYYMMDD_HH.sfc'
      write(*,*) ' or  '
      write(*,*) '   plot_winds.exe YYYYMMDDHHSW.dat'
      write(*,*) ' '
      stop
    endif 
    !
    ! === Step 2: read input file
   ! --- get dimension 
   open(1, file='dimension.dat', status='old')
     read(1,*) kx, ky
   close(1)

   ! --- read sfc_wind
   allocate( u(kx,ky), v(kx,ky) )

    len_file = len_trim(argum(1))
    open(1, file=trim(argum(1)), status='old', action='read' )
      read(1,*) lon_beg
      read(1,*) lat_beg
      read(1,*) res
      read(1,*)
      !
      read(1,*)
      read(1,'(15f7.2)') u(:,:)
      read(1,*)
      read(1,'(15f7.2)') v(:,:)
    close(1)
    !
    ! === Step 3: create data for GrADS plotting
    open(2,file='sfc_wind_tmp.dat',form='unformatted',access='direct',recl=4*kx*ky)
      write(2,rec=1) u
      write(2,rec=2) v
    close(2)
    !
    ! === Step 4: create CTL file for GrADS plotting
    open(9,file='sfc_wind_tmp.ctl',action='write')
      write(9,'(a)')   'dset sfc_wind_tmp.dat'
      write(9,'(a)')   'title SFC data'
      write(9,'(a)')   'undef -9999.'
      write(9,'(a,i6,a,2f12.4)')  'xdef ', kx, '  linear ', lon_beg, res
      write(9,'(a,i6,a,2f12.4)')  'ydef ', ky, '  linear ', lat_beg, res
      write(9,'(a)')       'zdef  1  linear  1    1'
      write(9,'(a)')       'tdef  1  linear  12:00Z11JAN2006 190MN'
      write(9,'(a)') 'vars  2'
      write(9,'(a)') 'u  0   99   sfc 10m AGL u wind (m/s)'
      write(9,'(a)') 'v  0   99   sfc 10m AGL v wind (m/s)'
      write(9,'(a)') 'endvars'
    close(9)
    !
    ! === Step 5: create GS file for GrADS plotting 
    open(9,file='sfc_wind_tmp.gs',action='write')
      write(9,'(a)')   "'open sfc_wind_tmp.ctl'"
      write(9,'(a)')   "'set grads off'"
      write(9,'(a)')   "'set mpdraw on'"
      write(9,'(a)')   "'set mpdset hires'"
      write(9,'(a)')   "'set map 8 1 6'"
      write(9,'(a)')   "'draw map '"
      if ( kx <= 100 ) then
        write(9,'(a)')   "'set xlint 1'"
      endif
      if ( ky <= 100 ) then
        write(9,'(a)')   "'set ylint 1'"
      endif
      write(9,'(a)')   "'set xlopts 1 6 0.16'"
      write(9,'(a)')   "'set ylopts 1 6 0.16'"
      write(9,'(a)')   "'set gxout shaded'"
      write(9,'(a)')   "'set clevs   10 12 14 16 18 20 22 24 26 28'"
      write(9,'(a)')   "'set ccols 0 11 5  13  3 10 7  12 8  2  6'"
      write(9,'(a)')   "'set cmin 10'"
      write(9,'(a)')   "'d mag(u,v)'"
      !write(9,'(a)')   "'cbarn'"
      write(9,'(a)')   "'cbarn 0.6 1'"
      write(9,'(a)')   "'set gxout vector'"
      write(9,'(a)')   "'set cmin 0'"
      write(9,'(a)')   "'set ccolor 1'"
      if( res <= 0.1 ) then
        write(9,'(a)')   "'d skip(u,15);skip(v,15)'"
      else
        write(9,'(a)')   "'d skip(u,5);skip(v,5)'"
      endif
      if( len_file <= 15 ) then
        write(9,'(a)')   "'draw title "//argum(1)(1:4)//"-"// &
                         argum(1)(5:6)//"-"//argum(1)(7:8)//"_"// &
                         argum(1)(10:11)//" UTC'"
      else if( len_file >= 16 ) then
        write(9,'(a)')   "'draw title "//argum(1)(1:4)//"-"// &
                         argum(1)(5:6)//"-"//argum(1)(7:8)//"_"// &
                         argum(1)(9:10)//" GMT (MM5)'"
      endif
      write(9,'(a)')   "'set strsiz 0.19'"
      write(9,'(a)')   "'set string 1 c 6'"
      if( res > 0.1 ) then
        !write(9,'(a)')   "'draw string 3.0 0.18 Maximum speed:"//max_spd//"'"
      else
        !write(9,'(a)')   "'draw string 3.8 0.18 Maximum speed:"//max_spd//"'"
      endif
      ! --- draw two sations
      !x1 = 93.70
      !y1 = 19.37
      !write(9,'(a,2(f7.2,a))')  "'q w2xy ",x1,"  ",y1, " '"
      !write(9,'(a)')  " x1 = subwrd(result,3)"
      !write(9,'(a)')  " y1 = subwrd(result,6)"
      !write(9,'(a)')  "'draw mark 3 'x1'  'y1' 0.13'"
      x2 = -14.00
      y2 =  10.00
      write(9,'(a,2(f7.2,a))')  "'q w2xy ",x2,"  ",y2, " '"
      write(9,'(a)')  " x2 = subwrd(result,3)"
      write(9,'(a)')  " y2 = subwrd(result,6)"
      write(9,'(a)')  "'draw mark 3 'x2'  'y2' 0.13'"
      !
      if( len_file <= 15 ) then
        write(9,'(a)')  "'enable print "//argum(1)(1:11)//".gm'"
        write(9,'(a)')  "'print'"
        write(9,'(a)')  "'disable print'"
        write(9,'(a)')  "'! gxeps -R -c -i "//argum(1)(1:11)//".gm -o "//argum(1)(1:11)//".eps'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:11)//".gm'"
        write(9,'(a)')  "'! convert -density 144 -resize 50% -antialias -trim "//argum(1)(1:11)//".eps "//argum(1)(1:11)//".bmp'"
        write(9,'(a)')  "'! convert -density 144 -antialias -trim "//argum(1)(1:11)//".bmp "//argum(1)(1:11)//".gif'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:11)//".bmp'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:11)//".eps'"
      else if( len_file >= 16 ) then
        write(9,'(a)')  "'enable print "//argum(1)(1:10)//".gm'"
        write(9,'(a)')  "'print'"
        write(9,'(a)')  "'disable print'"
        write(9,'(a)')  "'! gxeps -R -c -i "//argum(1)(1:10)//".gm -o "//argum(1)(1:10)//".eps'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:10)//".gm'"
        write(9,'(a)')  "'! convert -density 144 -antialias -trim "//argum(1)(1:10)//".eps "//argum(1)(1:10)//".bmp'"
        write(9,'(a)')  "'! convert -density 144 -antialias -trim "//argum(1)(1:10)//".bmp "//argum(1)(1:10)//".jpg'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:10)//".bmp'"
        write(9,'(a)')  "'! rm -f "//argum(1)(1:10)//".eps'"
      endif
      write(9,'(a)')   'quit'
    close(9)
    !
    ! === Step 6: do GrADS plotting
    !CALL system('grads -blc sfc_wind_tmp.gs > msg.tmp')
    !CALL system('rm -f sfc_wind_tmp.dat')
    !CALL system('rm -f sfc_wind_tmp.ctl')
    !CALL system('rm -f sfc_wind_tmp.gs msg.tmp')
    !
    !
  END PROGRAM plot_winds
