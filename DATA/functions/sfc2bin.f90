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
    character(len=8)   :: max_spd
    integer            :: kx, ky, len_file
    real, allocatable, dimension(:,:) :: u, v
    real               :: res, lon_beg, lat_beg
    !
    integer                           :: nargum, iargc, k
    character(len=100), dimension(10) :: argum
    !
    character(len=30)  :: temp_cha
    character(len=18)  :: sfc_year
    character(len=8)   :: sfc_data
    real               :: x1, x2, y1, y2
    integer            :: total_num,i
    character(len=100), allocatable, dimension(:) :: sfc_file

    !
    ! === Step 1: get input file
    nargum=iargc()
    do k = 1, nargum
      call getarg(k,argum(k))
    enddo
    !
    if( nargum == 0 ) then
      write(*,*) ' Usage:  '
   write(*,*) '   sfc2bin.exe year_ok year'
      write(*,*) '   sfc2bin.exe case_ok case1'
      write(*,*) ' '
      stop
    endif

    open(1,file=trim(argum(1)),status = 'old',action='read')
      read(1,*) total_num
      allocate ( sfc_file(total_num) )
      do i = 1 ,total_num
       read(1,'(a18,a50)') sfc_year,sfc_file(i)
      enddo
    close(1)
    !
    sfc_data=trim(sfc_file(1))
    ! === Step 2: read input file
    len_file = len_trim(argum(1))

    open(1, file=sfc_year//trim(sfc_file(1)), status='old',action='read' )
      read(1,*) temp_cha, temp_cha, date_time
      read(1,*) temp_cha, temp_cha, kx, temp_cha, ky
    close(1)

   open(2,file=trim(argum(2))//'_'//sfc_data//'.dat',access = 'direct',form = 'unformatted', recl = kx*ky*4)

    do i = 1,total_num
    print*,trim(sfc_file(i))
    open(1, file=sfc_year//trim(sfc_file(i)), status='old',action='read' )
      read(1,*) temp_cha, temp_cha, date_time
      read(1,*) temp_cha, temp_cha, kx, temp_cha, ky
      read(1,*) temp_cha, temp_cha, temp_cha, lon_beg
      read(1,*) temp_cha, temp_cha, temp_cha, lat_beg
      read(1,*) temp_cha, temp_cha, res
      read(1,*)
      read(1,*)
      read(1,'(a50)') speed
      read(1,*)
      max_spd = speed(13:17)//"m/s"
      !
      allocate( u(kx,ky), v(kx,ky) )
      read(1,*)
      read(1,'(15f5.1)') u(:,:)
      read(1,*)
      read(1,'(15f5.1)') v(:,:)
    close(1)

     !u = u / 1.15
     !v = v / 1.15

     write(2,rec = 2*i-1 ) u
     write(2,rec = 2*i ) v

  enddo
close(2)
    !
  END PROGRAM plot_winds

